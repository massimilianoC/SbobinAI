import importlib.util
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters import vad
from audio_transcript.adapters.vad import (
    EnergyDetector,
    SileroVadDetector,
    make_detector,
)
from audio_transcript.domain.models import TranscriptionError

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_ONNX = HAS_NUMPY and importlib.util.find_spec("onnxruntime") is not None
if HAS_NUMPY:
    # Import once, before any patch.dict(sys.modules, ...): a first import inside
    # such a block is removed again on exit, and NumPy 2 refuses to re-initialise
    # its C extension in the same process ("cannot load module more than once").
    import numpy  # noqa: F401
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
REAL_MODEL = Path(os.environ.get("SILERO_VAD_MODEL", "models/silero_vad.onnx"))


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def tone(seconds, rate=16000, amplitude=8000, frequency=220):
    return [
        int(amplitude * math.sin(2 * math.pi * frequency * i / rate))
        for i in range(int(seconds * rate))
    ]


def noise(seconds, rate=16000, amplitude=20, seed=1):
    rng = random.Random(seed)
    return [rng.randint(-amplitude, amplitude) for _ in range(int(seconds * rate))]


class MakeDetectorTests(unittest.TestCase):
    def build(self, name):
        return make_detector(
            name,
            model_path=Path("missing.onnx"),
            threshold=0.5,
            energy_margin_db=15.0,
            ffmpeg="ffmpeg",
        )

    def test_none_returns_no_detector(self):
        self.assertIsNone(self.build("none"))

    def test_known_names_build_detectors_without_importing_optional_runtime(self):
        with patch.dict(sys.modules, {"onnxruntime": None}):
            self.assertEqual(self.build("silero").name, "silero")
        self.assertEqual(self.build("energy").name, "energy")

    def test_unknown_name_is_actionable(self):
        with self.assertRaisesRegex(TranscriptionError, "Unknown speech detector 'webrtc'"):
            self.build("webrtc")


class SileroCheckTests(unittest.TestCase):
    def test_missing_model_points_to_install_script(self):
        detector = SileroVadDetector(Path("does-not-exist.onnx"))
        with self.assertRaisesRegex(TranscriptionError, "install-silero-vad.ps1"):
            detector.check()

    def test_missing_runtime_points_to_extra(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"x")
            with patch.dict(sys.modules, {"onnxruntime": None}):
                with self.assertRaisesRegex(TranscriptionError, r"pip install -e \.\[vad\]"):
                    SileroVadDetector(model).check()


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class SileroDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"fake")
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def fake_runtime(self, scripted):
        import numpy as np

        calls = self.calls

        class Session:
            def run(self, outputs, feeds):
                calls.append({key: np.array(value, copy=True) for key, value in feeds.items()})
                probability = scripted[min(len(calls) - 1, len(scripted) - 1)]
                state = feeds["state"] + 1.0
                return [np.array([[probability]], dtype=np.float32), state]

        class Options:
            pass

        created = {}

        def make_session(path, sess_options=None, providers=None):
            created.update(path=path, options=sess_options, providers=providers)
            return Session()

        runtime = types.SimpleNamespace(
            SessionOptions=Options, InferenceSession=make_session, created=created
        )
        return runtime

    def test_scripted_probabilities_context_state_and_padding(self):
        samples = [(i % 200) * 10 - 1000 for i in range(512 * 3 + 100)]
        wav_path = self.root / "audio.wav"
        write_wav(wav_path, samples)
        runtime = self.fake_runtime([0.1, 0.9, 0.2, 0.7])
        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            detector = SileroVadDetector(self.model, threshold=0.6)
            with patch.object(vad, "_READ_WINDOWS", 2):  # force several read blocks
                activity = detector.detect(wav_path)
        self.assertEqual(activity.probabilities, tuple(as_float32(p) for p in (0.1, 0.9, 0.2, 0.7)))
        self.assertAlmostEqual(activity.frame_seconds, 0.032)
        self.assertEqual(activity.threshold, 0.6)
        self.assertEqual(activity.detector, "silero")
        self.assertEqual(runtime.created["providers"], ["CPUExecutionProvider"])
        self.assertEqual(len(self.calls), 4)
        first, second, _, last = self.calls
        self.assertEqual(first["input"].shape, (1, 576))
        self.assertTrue((first["input"][0, :64] == 0).all())
        self.assertTrue((first["state"] == 0).all())
        self.assertTrue((second["state"] == 1).all())
        self.assertEqual(int(first["sr"]), 16000)
        # The context of each window is the last 64 samples of the previous model input.
        self.assertTrue((second["input"][0, :64] == first["input"][0, -64:]).all())
        # The final partial window is zero padded.
        self.assertTrue((last["input"][0, 64 + 100 :] == 0).all())
        self.assertAlmostEqual(float(last["input"][0, 64]), samples[1536] / 32768.0, places=6)

    def test_rejects_unsupported_audio_format(self):
        wav_path = self.root / "audio.wav"
        write_wav(wav_path, [0] * 800, rate=8000)
        with patch.dict(sys.modules, {"onnxruntime": self.fake_runtime([0.0])}):
            with self.assertRaisesRegex(TranscriptionError, "16 kHz mono"):
                SileroVadDetector(self.model).detect(wav_path)


def as_float32(value):
    import numpy as np

    return float(np.float32(value))


@unittest.skipUnless(HAS_FFMPEG, "FFmpeg not on PATH")
class EnergyDetectorTests(unittest.TestCase):
    def test_probabilities_map_the_adaptive_threshold_and_separate_speech_from_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "mix.wav"
            write_wav(wav_path, noise(2) + tone(2) + noise(2, seed=2))
            activity = EnergyDetector(threshold=0.5, margin_db=15).detect(wav_path)
        self.assertAlmostEqual(activity.frame_seconds, 0.03)
        self.assertAlmostEqual(len(activity.probabilities), 200, delta=2)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in activity.probabilities))
        self.assertEqual(activity.detector, "energy")
        for key in ("noise_floor_db", "threshold_db", "speech_level_db"):
            self.assertIn(key, activity.details)
        self.assertLess(activity.details["noise_floor_db"], activity.details["threshold_db"])
        quiet = activity.probabilities[10:60]
        loud = activity.probabilities[75:125]
        self.assertTrue(all(value < 0.35 for value in quiet))
        self.assertTrue(all(value > 0.5 for value in loud))

    def test_digital_silence_never_looks_like_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "silence.wav"
            write_wav(wav_path, [0] * 32000)
            activity = EnergyDetector().detect(wav_path)
        self.assertTrue(activity.probabilities)
        self.assertTrue(all(value < 0.5 for value in activity.probabilities))

    def test_missing_ffmpeg_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "silence.wav"
            write_wav(wav_path, [0] * 16000)
            with self.assertRaisesRegex(TranscriptionError, "not found"):
                EnergyDetector(ffmpeg="ffmpeg-does-not-exist").detect(wav_path)


class LevelMappingTests(unittest.TestCase):
    def test_threshold_level_maps_to_the_requested_probability(self):
        for threshold in (0.3, 0.5, 0.7):
            self.assertAlmostEqual(vad._level_probability(-40.0, -40.0, threshold), threshold)
            self.assertEqual(vad._level_probability(0.0, -40.0, threshold), 1.0)
            self.assertEqual(vad._level_probability(-90.0, -40.0, threshold), 0.0)
        self.assertLess(vad._level_probability(-43.0, -40.0, 0.5), 0.5 - 0.15 + 1e-9)


@unittest.skipUnless(HAS_ONNX and REAL_MODEL.is_file(), "Silero model or onnxruntime unavailable")
class RealSileroModelTests(unittest.TestCase):
    def test_silence_scores_low_and_output_is_framewise_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "silence.wav"
            write_wav(wav_path, noise(2, amplitude=5))
            activity = SileroVadDetector(REAL_MODEL).detect(wav_path)
        self.assertEqual(len(activity.probabilities), math.ceil(32000 / 512))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in activity.probabilities))
        self.assertLess(max(activity.probabilities), 0.5)


if __name__ == "__main__":
    unittest.main()

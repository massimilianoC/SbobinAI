import json
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from audio_transcript.adapters.ffmpeg import FFmpegProcessor
from audio_transcript.adapters.mock import MockBackend
from audio_transcript.adapters.nexa import NexaBackend
from audio_transcript.domain.models import AudioChunk, TranscriptionError


class FFmpegProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mp4"
        self.source.touch()
        self.processor = FFmpegProcessor("ffmpeg-test", "ffprobe-test")

    def tearDown(self):
        self.temp.cleanup()

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_probe_extracts_audio_metadata(self, run):
        run.return_value = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "12.5"},
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "sample_rate": "48000",
                            "channels": 2,
                        }
                    ],
                }
            ),
            stderr="",
        )
        info = self.processor.probe(self.source)
        self.assertEqual(
            (info.duration, info.sample_rate, info.channels, info.codec), (12.5, 48000, 2, "aac")
        )
        self.assertIsInstance(run.call_args.args[0], list)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_probe_rejects_media_without_audio(self, run):
        run.return_value = types.SimpleNamespace(
            returncode=0, stdout='{"format":{"duration":"2"},"streams":[]}', stderr=""
        )
        with self.assertRaisesRegex(TranscriptionError, "no audio"):
            self.processor.probe(self.source)

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_probe_rejects_malformed_metadata_actionably(self, run):
        run.return_value = types.SimpleNamespace(returncode=0, stdout='{"streams":null}', stderr="")
        with self.assertRaisesRegex(TranscriptionError, "invalid stream metadata"):
            self.processor.probe(self.source)

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_prepare_uses_bounded_segmented_pcm_and_actual_frame_times(self, run):
        def fake_run(command, **kwargs):
            if command[0] == "ffprobe-test":
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "format": {"duration": "1.5"},
                            "streams": [
                                {
                                    "codec_type": "audio",
                                    "codec_name": "aac",
                                    "sample_rate": "48000",
                                    "channels": 2,
                                }
                            ],
                        }
                    ),
                    stderr="",
                )
            output_pattern = Path(command[-1])
            for index, frames in enumerate((16000, 8000)):
                output = Path(str(output_pattern).replace("%08d", f"{index:08d}"))
                with wave.open(str(output), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(b"\0\0" * frames)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        run.side_effect = fake_run
        destination = self.root / "chunks"
        chunks = self.processor.prepare(
            self.source,
            destination,
            chunk_seconds=1,
            sample_rate=16000,
            max_duration=1.5,
        )
        self.assertEqual(
            [(chunk.start, chunk.end, chunk.index) for chunk in chunks],
            [
                (0.0, 1.0, 0),
                (1.0, 1.5, 1),
            ],
        )
        self.assertTrue(all(chunk.path.is_file() for chunk in chunks))
        command = run.call_args.args[0]
        filter_graph = command[command.index("-af") + 1]
        self.assertIn("atrim=end_sample=24000", filter_graph)
        self.assertIn("asetnsamples=n=16000:p=0", filter_graph)
        self.assertIn("segment", command)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    @patch("audio_transcript.adapters.ffmpeg.subprocess.run")
    def test_ffmpeg_failure_reports_diagnostic(self, run):
        run.return_value = types.SimpleNamespace(returncode=1, stdout="", stderr="decoder failed")
        with self.assertRaisesRegex(TranscriptionError, "decoder failed"):
            self.processor.probe(self.source)


class NexaBackendTests(unittest.TestCase):
    def test_check_imports_runtime_without_constructing_model(self):
        class FakeInference:
            def __init__(self, **kwargs):
                raise AssertionError("check must not initialize the model")

            def inference(self, audio_path, prompt=""):
                return "recognized speech"

            def cleanup(self):
                return None

        module = types.SimpleNamespace(NexaAudioLMInference=FakeInference)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            projector = Path(directory) / "projector.gguf"
            model.touch()
            projector.touch()
            backend = NexaBackend(local_path=model, projector_local_path=projector)
            with patch(
                "audio_transcript.adapters.nexa.importlib.import_module", return_value=module
            ):
                backend.check()

    def test_transcribe_loads_once_and_uses_chunk_bounds(self):
        created = []
        cleanups = []

        class FakeInference:
            def __init__(self, **kwargs):
                created.append(kwargs)

            def inference(self, audio_path, prompt=""):
                self.prompt = prompt
                return "  parole riconosciute  "

            def cleanup(self):
                cleanups.append(True)

        module = types.SimpleNamespace(NexaAudioLMInference=FakeInference)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            projector = Path(directory) / "projector.gguf"
            model.touch()
            projector.touch()
            backend = NexaBackend(
                model="qwen2audio",
                device="cpu",
                local_path=model,
                projector_local_path=projector,
            )
            chunk = AudioChunk(Path("chunk.wav"), start=4.5, end=7.0, index=2)
            with patch(
                "audio_transcript.adapters.nexa.importlib.import_module", return_value=module
            ):
                result = backend.transcribe(chunk, language="Italian")
                backend.transcribe(chunk, language="Italian")
            self.assertEqual(len(created), 1)
            self.assertEqual(
                created[0],
                {
                    "model_path": "qwen2audio",
                    "local_path": str(model),
                    "projector_local_path": str(projector),
                    "device": "cpu",
                },
            )
            self.assertEqual(len(cleanups), 2)
        self.assertEqual((result[0].start, result[0].end, result[0].speaker), (4.5, 7.0, None))
        self.assertEqual(result[0].text, "parole riconosciute")
        self.assertEqual(result[0].timing_source, "chunk")

    def test_unsupported_runtime_is_actionable(self):
        backend = NexaBackend()
        with patch(
            "audio_transcript.adapters.nexa.importlib.import_module",
            return_value=types.SimpleNamespace(),
        ):
            with self.assertRaisesRegex(TranscriptionError, "does not expose"):
                backend.check()

    def test_empty_response_is_not_reported_as_success(self):
        class FakeInference:
            def __init__(self, **kwargs):
                pass

            def inference(self, audio_path, prompt=""):
                return "  "

            def cleanup(self):
                return None

        module = types.SimpleNamespace(NexaAudioLMInference=FakeInference)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            projector = Path(directory) / "projector.gguf"
            model.touch()
            projector.touch()
            backend = NexaBackend(local_path=model, projector_local_path=projector)
            with patch(
                "audio_transcript.adapters.nexa.importlib.import_module", return_value=module
            ):
                with self.assertRaisesRegex(TranscriptionError, "empty transcript"):
                    backend.transcribe(AudioChunk(Path("x.wav"), 0, 1, 0), language=None)

    def test_missing_local_models_never_load_or_download(self):
        class FakeInference:
            def inference(self, audio_path, prompt=""):
                return ""

            def cleanup(self):
                return None

        backend = NexaBackend()
        module = types.SimpleNamespace(NexaAudioLMInference=FakeInference)
        with patch("audio_transcript.adapters.nexa.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(
                TranscriptionError, "Automatic Nexa model downloads are disabled"
            ):
                backend.check()

    def test_failed_inference_still_releases_native_context(self):
        cleanup_calls = []

        class FakeInference:
            def __init__(self, **kwargs):
                pass

            def inference(self, audio_path, prompt=""):
                raise RuntimeError("native inference failed")

            def cleanup(self):
                cleanup_calls.append(True)

        module = types.SimpleNamespace(NexaAudioLMInference=FakeInference)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            projector = Path(directory) / "projector.gguf"
            model.touch()
            projector.touch()
            backend = NexaBackend(local_path=model, projector_local_path=projector)
            with patch(
                "audio_transcript.adapters.nexa.importlib.import_module", return_value=module
            ):
                with self.assertRaisesRegex(TranscriptionError, "native inference failed"):
                    backend.transcribe(AudioChunk(Path("x.wav"), 0, 1, 0), language=None)
        self.assertEqual(cleanup_calls, [True])


class MockBackendTests(unittest.TestCase):
    def test_emits_explicit_synthetic_diagnostic(self):
        backend = MockBackend()
        segment = backend.transcribe(AudioChunk(Path("x.wav"), 1, 2, 0), language=None)[0]
        self.assertIn("MOCK TRANSCRIPT", segment.text)
        self.assertIn("no speech recognition performed", segment.text)
        self.assertIsNone(segment.speaker)
        backend.check()
        backend.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from audio_transcript.cli import main
from audio_transcript.config import load_config

ROOT = Path(__file__).resolve().parents[1]

BASE = """
[audio_transcript]
backend = "llamacpp"
model = "demo-model"
base_url = "http://127.0.0.1:8088"
parallel_requests = 2
vad_model_path = "silero-vad/silero_vad.onnx"
"""

TABLES = """
[resources]
store = "{store}"

[server]
runtime_dir = "runtimes/llama.cpp-cuda"
model_path = "Demo/model.gguf"
projector_path = "Demo/mmproj.gguf"
alias = "demo-model"
state_dir = "services/demo"
port = 8088
context_size = 8192
parallel = 2
"""


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = self.tmp / "store"

    def write(self, text: str) -> Path:
        path = self.tmp / "config.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def full(self, **replace: str) -> str:
        text = BASE + TABLES.format(store=self.store.as_posix())
        for old, new in replace.items():
            key = old.replace("__", " ")
            self.assertIn(key, text)
            text = text.replace(key, new)
        return text


class ServerConfigTests(_Base):
    def test_relative_paths_resolve_against_the_store(self):
        config = load_config(self.write(self.full()))
        server = config.server
        self.assertEqual(config.resources.store, self.store.resolve())
        self.assertEqual(server.model_path, (self.store / "Demo" / "model.gguf").resolve())
        self.assertEqual(server.runtime_dir, (self.store / "runtimes" / "llama.cpp-cuda").resolve())
        self.assertEqual(server.state_dir, (self.store / "services" / "demo").resolve())
        self.assertEqual(config.vad_model_path, self.store.resolve() / "silero-vad/silero_vad.onnx")
        self.assertIsNone(server.gpu_layers)

    def test_absolute_paths_are_kept_and_gpu_layers_read(self):
        absolute = (self.tmp / "elsewhere" / "model.gguf").resolve()
        text = self.full(**{'model_path__=__"Demo/model.gguf"': f"model_path = '{absolute}'"})
        text += "gpu_layers = 40\n"
        config = load_config(self.write(text))
        self.assertEqual(config.server.model_path, absolute)
        self.assertEqual(config.server.gpu_layers, 40)

    def test_without_the_new_tables_the_config_still_works(self):
        config = load_config(self.write(BASE))
        self.assertIsNone(config.server)
        self.assertIsNone(config.resources)
        self.assertEqual(config.vad_model_path, Path("silero-vad/silero_vad.onnx"))

    def test_flat_legacy_file_still_loads(self):
        config = load_config(self.write('backend = "mock"\n'))
        self.assertEqual(config.backend, "mock")

    def test_invalid_server_settings_are_rejected(self):
        cases = {
            "alias differs": self.full(**{'alias__=__"demo-model"': 'alias = "other"'}),
            "port mismatch": self.full(**{"port__=__8088": "port = 9000"}),
            "parallel too low": self.full(**{"parallel__=__2": "parallel = 1"}),
            "port range": self.full(**{"port__=__8088": "port = 70000"}),
            "bool as int": self.full(**{"context_size__=__8192": "context_size = true"}),
            "unknown key": self.full() + "colour = 1\n",
            "remote base_url": self.full(
                **{'base_url__=__"http://127.0.0.1:8088"': 'base_url = "http://example.com:8088"'}
            ),
            "wrong backend": self.full(**{'backend__=__"llamacpp"': 'backend = "mock"'}),
        }
        for label, text in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                load_config(self.write(text))

    def test_missing_server_key_names_the_key(self):
        text = self.full().replace('alias = "demo-model"\n', "")
        with self.assertRaisesRegex(ValueError, "alias"):
            load_config(self.write(text))

    def test_relative_server_path_needs_a_store(self):
        text = BASE + TABLES.format(store="x").replace('[resources]\nstore = "x"\n', "")
        with self.assertRaisesRegex(ValueError, "resources"):
            load_config(self.write(text))

    def test_unknown_tables_and_misplaced_tables_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration table"):
            load_config(self.write(BASE + "\n[extra]\nx = 1\n"))
        with self.assertRaisesRegex(ValueError, "Unknown \\[resources\\]"):
            load_config(self.write(BASE + '\n[resources]\nstore = "x"\nother = 1\n'))
        with self.assertRaisesRegex(ValueError, "store"):
            load_config(self.write(BASE + "\n[resources]\n"))

    def test_example_config_is_valid(self):
        config = load_config(ROOT / "config.example.toml")
        self.assertEqual(config.server.alias, config.model)
        self.assertGreaterEqual(config.server.parallel, config.parallel_requests)


class ServerCliTests(_Base):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_server_profile_prints_absolute_json(self):
        cpu = self.store / "runtimes" / "llama.cpp-cpu-b11389"
        cpu.mkdir(parents=True)
        (cpu / "llama-server.exe").write_bytes(b"")  # only the CPU runtime: no device probe
        path = self.write(self.full())
        code, out, _ = self.run_cli(["server-profile", "--config", str(path)])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["alias"], "demo-model")
        self.assertEqual(data["port"], 8088)
        self.assertEqual(data["parallel"], 2)
        self.assertIsNone(data["gpu_layers"])
        for key in ("runtime_dir", "model_path", "projector_path", "state_dir"):
            self.assertTrue(Path(data[key]).is_absolute(), key)
        self.assertEqual(Path(data["model_path"]), (self.store / "Demo" / "model.gguf").resolve())

    def test_server_profile_without_table_exits_2_or_null_when_optional(self):
        path = self.write(BASE)
        code, _, err = self.run_cli(["server-profile", "--config", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("[server]", err)
        code, out, _ = self.run_cli(["server-profile", "--config", str(path), "--optional"])
        self.assertEqual((code, out.strip()), (0, "null"))

    def test_server_profile_with_invalid_table_exits_2_even_when_optional(self):
        path = self.write(self.full(**{"port__=__8088": "port = 9000"}))
        code, _, err = self.run_cli(["server-profile", "--config", str(path), "--optional"])
        self.assertEqual(code, 2)
        self.assertIn("port", err)

    def test_empty_queue_prints_a_hint_and_exits_zero(self):
        path = self.write(
            f'[audio_transcript]\nbackend = "mock"\nvad = "none"\n'
            f'input_dir = "{(self.tmp / "in").as_posix()}"\n'
            f'process_dir = "{(self.tmp / "pr").as_posix()}"\n'
            f'output_dir = "{(self.tmp / "out").as_posix()}"\n'
            f'processed_dir = "{(self.tmp / "done").as_posix()}"\n'
        )
        code, out, _ = self.run_cli(["run", "--config", str(path)])
        self.assertEqual(code, 0)
        self.assertIn(
            f"No media files found in {self.tmp / 'in'}; "
            "put audio/video files there and run again.",
            out,
        )


if __name__ == "__main__":
    unittest.main()

"""Inference backend selection (CUDA, Vulkan, CPU), setup of runtime archives and provenance.

Everything is synthetic: fake runtime folders, a fake ``--list-devices`` probe and fake ZIP
archives. No llama-server is started and nothing is downloaded.
"""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audio_transcript import cli
from audio_transcript.adapters import devices as devices_module
from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application import backend_select as bs
from audio_transcript.application import resources as res
from audio_transcript.application.events import CallbackSink, EventBus
from audio_transcript.application.pipeline import TranscriptionPipeline
from audio_transcript.application.progress import RunState
from audio_transcript.application.session import RunSession
from audio_transcript.config import DEFAULT_RUNTIME_FOLDERS, load_config
from audio_transcript.ui.live import render

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config

ROOT = Path(__file__).resolve().parents[1]

CUDA_LISTING = (
    "Available devices:\n  CUDA0: NVIDIA GeForce RTX 5070 Ti (16302 MiB, 15039 MiB free)\n"
)
VULKAN_LISTING = (
    "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 5070 Ti (16211 MiB, 15429 MiB free)\n"
)
NONE_LISTING = "Available devices:\n  (none)\n"

SERVER_TABLES = """
[audio_transcript]
backend = "llamacpp"
model = "demo-model"
base_url = "http://127.0.0.1:8088"
vad = "none"

[resources]
store = "{store}"

[server]
model_path = "Demo/model.gguf"
projector_path = "Demo/mmproj.gguf"
alias = "demo-model"
state_dir = "services/demo"
port = 8088
context_size = 8192
parallel = 2
"""


def make_runtime(folder: Path, backend: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "llama-server.exe").write_bytes(b"")
    if backend in {"cuda", "vulkan"}:
        (folder / f"ggml-{backend}.dll").write_bytes(b"")
    return folder


class FakeProbe:
    """Stands in for ``llama-server --list-devices``; counts calls per runtime folder."""

    def __init__(
        self, listings: dict[str, str] | None = None, errors: dict[str, str] | None = None
    ):
        self.listings = listings or {}
        self.errors = errors or {}
        self.calls: list[str] = []

    def __call__(self, folder: Path) -> bs.DeviceListing:
        self.calls.append(folder.name)
        if folder.name in self.errors:
            return bs.DeviceListing(error=self.errors[folder.name])
        return bs.DeviceListing(devices=bs.parse_devices(self.listings.get(folder.name, "")))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def server(self, backends=("cuda", "vulkan", "cpu"), **options):
        runtimes = {name: make_runtime(self.root / name, name) for name in backends}
        for name in ("cuda", "vulkan", "cpu"):
            runtimes.setdefault(name, self.root / name)  # configured but not installed
        values = {
            "runtime_dir": runtimes["cuda"],
            "model_path": Path("m.gguf"),
            "projector_path": Path("p.gguf"),
            "alias": "demo-model",
            "state_dir": self.root / "state",
            "port": 8088,
            "context_size": 8192,
            "parallel": 2,
            "runtimes": runtimes,
        }
        values.update(options)
        return SimpleNamespace(
            **{
                "backend": "auto",
                "fallback": ("cuda", "vulkan", "cpu"),
                "device": None,
                "threads": None,
                **values,
            }
        )

    def test_cuda_present_selects_cuda_without_fallback(self):
        probe = FakeProbe({"cuda": CUDA_LISTING, "vulkan": VULKAN_LISTING})
        selection = bs.select_backend(self.server(), probe, threads_default=lambda: 6)
        self.assertEqual((selection.backend, selection.device), ("cuda", "CUDA0"))
        self.assertEqual(selection.device_name, "NVIDIA GeForce RTX 5070 Ti")
        self.assertFalse(selection.fallback_used)
        self.assertIsNone(selection.warning)
        self.assertEqual(probe.calls, ["cuda"])  # later candidates are not probed
        self.assertEqual(bs.warning_lines(selection), [])

    def test_no_cuda_device_falls_back_to_vulkan_with_reasons(self):
        probe = FakeProbe({"cuda": NONE_LISTING, "vulkan": VULKAN_LISTING})
        selection = bs.select_backend(self.server(), probe)
        self.assertEqual((selection.backend, selection.device), ("vulkan", "Vulkan0"))
        self.assertTrue(selection.fallback_used)
        self.assertEqual(
            [(a.backend, a.ok) for a in selection.tried], [("cuda", False), ("vulkan", True)]
        )
        self.assertIn("no CUDA device", selection.tried[0].reason)
        self.assertIn("about 2x slower", selection.warning)
        self.assertIn("skipped cuda", "\n".join(bs.warning_lines(selection)))
        self.assertEqual(
            bs.describe_runtime(selection.run_info()),
            "Backend: vulkan Vulkan0 NVIDIA GeForce RTX 5070 Ti "
            "(fallback from cuda: no CUDA device reported)",
        )

    def test_missing_cuda_runtime_and_dll_are_reported(self):
        probe = FakeProbe({"vulkan": VULKAN_LISTING})
        selection = bs.select_backend(self.server(("vulkan", "cpu")), probe)
        self.assertEqual(selection.backend, "vulkan")
        self.assertIn("runtime not installed", selection.tried[0].reason)
        self.assertEqual(probe.calls, ["vulkan"])
        server = self.server(("cuda", "vulkan", "cpu"))
        (server.runtimes["cuda"] / "ggml-cuda.dll").unlink()
        selection = bs.select_backend(server, FakeProbe({"vulkan": VULKAN_LISTING}))
        self.assertEqual(selection.backend, "vulkan")
        self.assertEqual(selection.tried[0].reason, "ggml-cuda.dll missing")

    def test_no_gpu_at_all_selects_cpu_with_threads(self):
        probe = FakeProbe({"cuda": NONE_LISTING, "vulkan": NONE_LISTING})
        selection = bs.select_backend(self.server(), probe, threads_default=lambda: 6)
        self.assertEqual((selection.backend, selection.device), ("cpu", "none"))
        self.assertEqual(selection.threads, 6)
        self.assertTrue(selection.fallback_used)
        self.assertIn("about 9x slower", selection.warning)
        configured = bs.select_backend(self.server(threads=3), probe)
        self.assertEqual(configured.threads, 3)
        self.assertEqual(bs.header_label(selection.run_info()), "CPU (fallback)")

    def test_cpu_reuses_another_runtime_when_no_cpu_runtime_is_installed(self):
        probe = FakeProbe({"cuda": NONE_LISTING, "vulkan": NONE_LISTING})
        selection = bs.select_backend(self.server(("cuda", "vulkan")), probe)
        self.assertEqual(selection.backend, "cpu")
        self.assertEqual(selection.runtime_dir.name, "cuda")
        self.assertIn("using the CPU path", selection.tried[-1].reason)

    def test_software_vulkan_renderer_is_not_a_gpu(self):
        listing = "Available devices:\n  Vulkan0: llvmpipe (LLVM 17.0.6, 256 bits) (8000 MiB)\n"
        selection = bs.select_backend(
            self.server(), FakeProbe({"cuda": NONE_LISTING, "vulkan": listing})
        )
        self.assertEqual(selection.backend, "cpu")
        self.assertIn("software", selection.tried[1].reason)

    def test_fixed_backend_never_falls_back(self):
        probe = FakeProbe({"cuda": NONE_LISTING, "vulkan": VULKAN_LISTING})
        with self.assertRaises(bs.BackendSelectionError) as caught:
            bs.select_backend(self.server(backend="cuda"), probe)
        message = str(caught.exception)
        self.assertIn('backend = "cuda"', message)
        self.assertIn("never falls back", message)
        self.assertEqual(probe.calls, ["cuda"])
        ok = bs.select_backend(self.server(backend="vulkan"), probe)
        self.assertEqual(ok.backend, "vulkan")
        self.assertFalse(ok.fallback_used)
        cpu = bs.select_backend(self.server(backend="cpu"), FakeProbe())
        self.assertEqual(cpu.backend, "cpu")
        self.assertFalse(cpu.fallback_used)

    def test_nothing_usable_is_a_clear_error(self):
        with self.assertRaisesRegex(bs.BackendSelectionError, "No usable inference backend"):
            bs.select_backend(self.server(()), FakeProbe())

    def test_fallback_order_is_configurable_and_first_choice_is_no_fallback(self):
        probe = FakeProbe({"cuda": CUDA_LISTING, "vulkan": VULKAN_LISTING})
        selection = bs.select_backend(self.server(fallback=("vulkan", "cuda", "cpu")), probe)
        self.assertEqual(selection.backend, "vulkan")
        self.assertFalse(selection.fallback_used)

    def test_list_devices_timeout_or_failure_is_handled(self):
        probe = FakeProbe(
            {"vulkan": VULKAN_LISTING}, errors={"cuda": "--list-devices timed out after 30 s"}
        )
        selection = bs.select_backend(self.server(), probe)
        self.assertEqual(selection.backend, "vulkan")
        self.assertIn("timed out", selection.tried[0].reason)

    def test_real_probe_reports_timeout_and_missing_executable(self):
        folder = make_runtime(self.root / "rt", "cuda")
        self.assertEqual(
            devices_module.list_devices(self.root / "nowhere").error,
            "llama-server executable missing",
        )
        import subprocess

        with patch.object(
            devices_module.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("llama-server", 1),
        ):
            listing = devices_module.list_devices(folder, timeout=1)
        self.assertIn("timed out after 1 s", listing.error)
        completed = SimpleNamespace(returncode=0, stdout=CUDA_LISTING, stderr="")
        with patch.object(devices_module.subprocess, "run", return_value=completed):
            listing = devices_module.list_devices(folder)
        self.assertEqual(listing.devices[0].id, "CUDA0")

    def test_device_choice_is_honoured_and_validated(self):
        listing = (
            "Available devices:\n  Vulkan0: Intel(R) UHD Graphics (8000 MiB, 7000 MiB free)\n"
            "  Vulkan1: AMD Radeon RX 7800 XT (16000 MiB, 15000 MiB free)\n"
        )
        probe = FakeProbe({"cuda": NONE_LISTING, "vulkan": listing})
        selection = bs.select_backend(self.server(device="Vulkan1"), probe)
        self.assertEqual(
            (selection.device, selection.device_name), ("Vulkan1", "AMD Radeon RX 7800 XT")
        )
        with self.assertRaises(bs.BackendSelectionError):
            bs.select_backend(self.server(device="Vulkan3", backend="vulkan"), probe)

    def test_parse_devices_ignores_log_noise_and_memory_suffix(self):
        text = (
            "0.00.001.085 I srv  llama_server: initializing ...\n"
            "ggml_vulkan: Found 1 Vulkan devices:\n"
            "Available devices:\n"
            "  Vulkan0: NVIDIA GeForce RTX 5070 Ti (16211 MiB, 15429 MiB free)\n"
        )
        devices = bs.parse_devices(text)
        self.assertEqual(
            [(d.id, d.description) for d in devices], [("Vulkan0", "NVIDIA GeForce RTX 5070 Ti")]
        )
        self.assertEqual(bs.parse_devices(NONE_LISTING), ())

    def test_run_info_and_dict_do_not_leak_paths_into_provenance(self):
        selection = bs.select_backend(
            self.server(), FakeProbe({"cuda": CUDA_LISTING}), threads_default=lambda: 4
        )
        info = selection.run_info()
        self.assertNotIn(str(self.root), json.dumps(info))
        self.assertEqual(
            set(info),
            {
                "backend",
                "device",
                "device_name",
                "fallback_used",
                "requested",
                "threads",
                "runtime_tag",
                "skipped",
            },
        )
        self.assertIn(str(self.root), selection.to_dict()["runtime_dir"])


class ServerConfigBackendTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = self.tmp / "store"

    def load(self, extra: str = "", *, runtime: str | None = 'runtime_dir = "rt/cuda"\n'):
        text = SERVER_TABLES.format(store=self.store.as_posix())
        text = text.replace("[server]\n", "[server]\n" + (runtime or ""), 1) + extra
        path = self.tmp / "config.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def test_runtime_dir_only_stays_the_cuda_runtime(self):
        server = self.load().server
        self.assertEqual(server.backend, "auto")
        self.assertEqual(server.fallback, ("cuda", "vulkan", "cpu"))
        self.assertEqual(server.runtimes["cuda"], (self.store / "rt" / "cuda").resolve())
        self.assertEqual(server.runtime_dir, server.runtimes["cuda"])
        # Missing backends default to the folders `setup` installs into.
        self.assertEqual(
            server.runtimes["vulkan"], (self.store / DEFAULT_RUNTIME_FOLDERS["vulkan"]).resolve()
        )
        self.assertIsNone(server.device)
        self.assertIsNone(server.threads)

    def test_runtimes_table_and_options(self):
        server = self.load(
            'backend = "cpu"\nfallback = ["cpu"]\nthreads = 4\n'
            "\n[server.runtimes]\ncpu = 'cpu-folder'\nvulkan = 'vk'\n",
            runtime=None,
        ).server
        self.assertEqual(server.backend, "cpu")
        self.assertEqual(server.fallback, ("cpu",))
        self.assertEqual(server.threads, 4)
        self.assertEqual(server.runtimes["cpu"], (self.store / "cpu-folder").resolve())
        self.assertEqual(server.runtime_dir, server.runtimes["cuda"])

    def test_invalid_backend_settings_have_actionable_errors(self):
        cases = {
            'backend = "metal"\n': "backend must be",
            "fallback = []\n": "fallback must be",
            'fallback = ["cuda", "cuda"]\n': "distinct",
            'fallback = ["gpu"]\n': "fallback must be",
            'device = "Vulkan"\n': "device must look like",
            'backend = "cuda"\ndevice = "Vulkan0"\n': "belongs to the vulkan backend",
            'backend = "auto"\nfallback = ["cpu"]\ndevice = "CUDA0"\n': "not in",
            "threads = 0\n": "threads",
            "threads = true\n": "threads",
            "\n[server.runtimes]\nrocm = 'x'\n": r"Unknown \[server.runtimes\]",
        }
        for extra, message in cases.items():
            # keys must precede the [server.runtimes] sub-table
            with self.subTest(extra), self.assertRaisesRegex(ValueError, message):
                self.load(extra)

    def test_runtime_is_required_and_relative_needs_store(self):
        text = SERVER_TABLES.format(store="x").replace('[resources]\nstore = "x"\n', "")
        for name in ("model_path", "projector_path", "state_dir"):
            text = text.replace(f'{name} = "', f'{name} = "{self.tmp.as_posix()}/')
        path = self.tmp / "nostore.toml"
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"runtime_dir or a \[server.runtimes\]"):
            load_config(path)
        path.write_text(
            text.replace("[server]\n", '[server]\nruntime_dir = "relative/cuda"\n', 1),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "relative"):
            load_config(path)

    def test_default_runtime_folders_match_the_manifest(self):
        manifest = res.load_manifest(ROOT / "resources.json")
        destinations = {
            r.backend: r.destination
            for r in manifest.resources.values()
            if r.kind == "runtime" and r.backend
        }
        self.assertEqual(destinations["vulkan"], DEFAULT_RUNTIME_FOLDERS["vulkan"])
        self.assertEqual(destinations["cpu"], DEFAULT_RUNTIME_FOLDERS["cpu"])
        self.assertEqual(destinations["cuda"], DEFAULT_RUNTIME_FOLDERS["cuda"])

    def test_example_config_describes_the_backends(self):
        config = load_config(ROOT / "config.example.toml")
        self.assertEqual(config.server.backend, "auto")
        self.assertEqual(set(config.server.runtimes), {"cuda", "vulkan", "cpu"})

    def test_backend_keys_are_not_part_of_the_job_fingerprint(self):
        def fingerprint(extra: str) -> str:
            return TranscriptionPipeline._fingerprint(SimpleNamespace(config=self.load(extra)))

        reference = fingerprint("")
        self.assertEqual(reference, fingerprint('backend = "cpu"\nthreads = 3\n'))
        self.assertEqual(reference, fingerprint('backend = "vulkan"\ndevice = "Vulkan1"\n'))


def _fake_zip(path: Path, files: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeManifestTests(unittest.TestCase):
    def test_variants_are_pinned_and_well_formed(self):
        manifest = res.load_manifest(ROOT / "resources.json")
        variants = {
            r.backend: r for r in manifest.resources.values() if r.kind == "runtime" and r.install
        }
        self.assertEqual(set(variants), {"vulkan", "cpu"})
        vulkan, cpu = variants["vulkan"], variants["cpu"]
        self.assertEqual(vulkan.id, "llamacpp-b11389-win-x64-vulkan")
        self.assertEqual(vulkan.size_bytes, 33279915)
        self.assertEqual(
            vulkan.sha256, "7bddec331a6f3328667d8476ee092a3747493ecc966b22e6cc64c0dc1aebedc9"
        )
        self.assertEqual(cpu.size_bytes, 19363530)
        self.assertEqual(
            cpu.sha256, "32ff4a22592737b322b56176be91c1802b607611bc1c65329ef167140aa61c17"
        )
        for variant in variants.values():
            self.assertEqual(variant.revision, "b11389")
            self.assertTrue(
                variant.url.startswith(
                    "https://github.com/ggml-org/llama.cpp/releases/download/b11389/"
                )
            )
            self.assertTrue(variant.url.endswith(f"-bin-win-{variant.backend}-x64.zip"))
            self.assertEqual(variant.license, "MIT (ggml-org/llama.cpp)")
            self.assertEqual(variant.destination, f"runtimes/llama.cpp-{variant.backend}-b11389")
            self.assertRegex(variant.sha256, r"^[0-9a-f]{64}$")
        cuda = [r for r in manifest.resources.values() if r.backend == "cuda"]
        self.assertEqual(len(cuda), 2)
        self.assertTrue(all(r.revision == "b11193" and not r.install for r in cuda))
        for profile in manifest.profiles.values():
            backends = {
                manifest.resources[rid].backend
                for rid in profile.resources
                if manifest.resources[rid].kind == "runtime"
            }
            self.assertEqual(backends, {"cuda", "vulkan", "cpu"})

    def test_bad_backend_or_install_fields_are_rejected(self):
        data = json.loads((ROOT / "resources.json").read_text(encoding="utf-8"))
        for resource_id, change in (
            ("llamacpp-b11389-win-x64-cpu", {"backend": "rocm"}),
            ("llamacpp-b11389-win-x64-cpu", {"install": "tar"}),
            ("llamacpp-b11389-win-x64-cpu", {"url": "https://example.com/x.exe"}),
            ("silero-vad-onnx", {"backend": "cpu"}),
        ):
            broken = json.loads(json.dumps(data))
            for entry in broken["resources"]:
                if entry["id"] == resource_id:
                    entry.update(change)
            with self.subTest(change), self.assertRaises(ValueError):
                res.parse_manifest(broken)


class InstallRuntimeTests(unittest.TestCase):
    FILES = {
        "llama-server.exe": b"MZ fake server",
        "ggml-vulkan.dll": b"fake dll",
        "licenses/LICENSE.txt": b"MIT",
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Path(self._tmp.name) / "store"
        self.zip = Path(self._tmp.name) / "asset.zip"
        digest = _fake_zip(self.zip, self.FILES)
        self.resource = res.Resource(
            id="rt-vulkan",
            kind="runtime",
            platforms=("any",),
            url="https://example.invalid/releases/download/b1/llama-b1-bin-win-vulkan-x64.zip",
            revision="b1",
            sha256=digest,
            size_bytes=self.zip.stat().st_size,
            license="MIT",
            source="https://example.invalid/releases/tag/b1",
            destination="runtimes/llama.cpp-vulkan-b1",
            backend="vulkan",
            install="zip",
        )
        self.fetched = 0

    def fetch(self, url, destination, *, sha256, size_bytes, progress=None):
        self.fetched += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.zip.read_bytes())

    def test_extracts_atomically_and_writes_manifest_and_provenance(self):
        folder = res.install_runtime(
            self.resource,
            self.store,
            fetch=self.fetch,
            hasher=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        self.assertEqual(folder, self.store / "runtimes" / "llama.cpp-vulkan-b1")
        for name in self.FILES:
            self.assertEqual((folder / name).read_bytes(), self.FILES[name])
        manifest = json.loads((folder / "runtime-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tag"], "b1")
        self.assertEqual(manifest["backend"], "vulkan")
        self.assertEqual(manifest["asset"], "llama-b1-bin-win-vulkan-x64.zip")
        self.assertEqual(manifest["sha256"], self.resource.sha256)
        self.assertEqual(manifest["files"], sorted(self.FILES))
        self.assertIn("Backend:  vulkan", (folder / "PROVENANCE.txt").read_text(encoding="utf-8"))
        self.assertTrue(res.runtime_installed(folder, self.resource))
        leftovers = [p.name for p in folder.parent.iterdir() if p.name != folder.name]
        self.assertEqual(leftovers, [res.DOWNLOAD_FOLDER])  # archive removed, no temp folders
        self.assertEqual(list((folder.parent / res.DOWNLOAD_FOLDER).iterdir()), [])

    def test_verified_installation_is_skipped(self):
        hasher = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        res.install_runtime(self.resource, self.store, fetch=self.fetch, hasher=hasher)
        res.install_runtime(self.resource, self.store, fetch=self.fetch, hasher=hasher)
        self.assertEqual(self.fetched, 1)
        item = res.build_plan(
            SimpleNamespace(resources={"rt-vulkan": self.resource}),
            SimpleNamespace(resources=("rt-vulkan",)),
            self.store,
            platform_id="windows-x64",
            hasher=hasher,
        )[0]
        self.assertEqual(item.status, "installed")

    def test_tampered_archive_is_refused_and_nothing_is_installed(self):
        def tampered(url, destination, *, sha256, size_bytes, progress=None):
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = bytearray(self.zip.read_bytes())
            data[-30] ^= 0xFF
            destination.write_bytes(bytes(data))

        with self.assertRaisesRegex(res.RuntimeInstallError, "does not match the pinned SHA-256"):
            res.install_runtime(
                self.resource,
                self.store,
                fetch=tampered,
                hasher=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        self.assertFalse((self.store / "runtimes" / "llama.cpp-vulkan-b1").exists())

    def test_unsafe_member_and_foreign_folder_are_refused(self):
        evil = Path(self._tmp.name) / "evil.zip"
        digest = _fake_zip(evil, {"../escape.txt": b"x"})
        resource = res.Resource(
            **{**self.resource.__dict__, "sha256": digest, "size_bytes": evil.stat().st_size}
        )
        with self.assertRaisesRegex(res.RuntimeInstallError, "Unsafe path"):
            res.extract_runtime_archive(evil, resource, self.store / "runtimes" / "x")
        self.assertEqual(
            [p.name for p in (self.store / "runtimes").iterdir()], []
        )  # scratch folder cleaned up
        foreign = self.store / "runtimes" / "llama.cpp-vulkan-b1"
        foreign.mkdir(parents=True)
        (foreign / "keep.txt").write_text("mine", encoding="utf-8")
        with self.assertRaisesRegex(res.RuntimeInstallError, "never overwritten"):
            res.extract_runtime_archive(self.zip, self.resource, foreign)
        self.assertEqual((foreign / "keep.txt").read_text(encoding="utf-8"), "mine")
        item = res.build_plan(
            SimpleNamespace(resources={"rt-vulkan": self.resource}),
            SimpleNamespace(resources=("rt-vulkan",)),
            self.store,
            platform_id="windows-x64",
            hasher=lambda p: "",
        )[0]
        self.assertEqual(item.status, "conflict")

    def test_not_a_zip_is_reported(self):
        junk = Path(self._tmp.name) / "junk.zip"
        junk.write_bytes(b"not a zip")
        with self.assertRaisesRegex(res.RuntimeInstallError, "not a valid ZIP"):
            res.extract_runtime_archive(junk, self.resource, self.store / "runtimes" / "j")


class SetupBackendsCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = self.tmp / "store"
        archive = self.tmp / "asset.zip"
        digest = _fake_zip(archive, {"llama-server.exe": b"x", "ggml-cpu-x64.dll": b"y"})
        self.archive = archive
        runtime = {
            "kind": "runtime",
            "platforms": ["any"],
            "license": "MIT",
            "source": "https://example.invalid/tag",
            "revision": "b1",
        }
        self.manifest = self.tmp / "resources.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "default_profile": "p",
                    "resources": [
                        {
                            "id": "rt-cpu",
                            "backend": "cpu",
                            "install": "zip",
                            "url": "https://example.invalid/b1/llama-b1-bin-win-cpu-x64.zip",
                            "sha256": digest,
                            "size_bytes": archive.stat().st_size,
                            "destination": "runtimes/llama.cpp-cpu-b1",
                            **runtime,
                        },
                        {
                            "id": "rt-vulkan",
                            "backend": "vulkan",
                            "install": "zip",
                            "url": "https://example.invalid/b1/llama-b1-bin-win-vulkan-x64.zip",
                            "sha256": digest,
                            "size_bytes": archive.stat().st_size,
                            "destination": "runtimes/llama.cpp-vulkan-b1",
                            **runtime,
                        },
                        {
                            "id": "rt-cuda",
                            "backend": "cuda",
                            "url": "https://example.invalid/b1/llama-b1-bin-win-cuda-x64.zip",
                            "sha256": digest,
                            "size_bytes": 1_000_000,
                            "destination": "runtimes/llama.cpp-cuda",
                            **runtime,
                        },
                    ],
                    "profiles": {"p": {"resources": ["rt-cuda", "rt-vulkan", "rt-cpu"]}},
                }
            ),
            encoding="utf-8",
        )

    def fetch(self, url, destination, *, sha256, size_bytes, progress=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.archive.read_bytes())

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(
                ["setup", "--manifest", str(self.manifest), "--store", str(self.store), *argv]
            )
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_shows_each_backend_and_size(self):
        code, out, _ = self.run_cli("--dry-run", "--backends", "cuda,vulkan,cpu")
        self.assertEqual(code, 0)
        for backend in ("cuda", "vulkan", "cpu"):
            self.assertRegex(out, rf"  {backend}: \d")
        self.assertIn("install-llamacpp-cuda.ps1", out)
        self.assertIn("to download and install", out)
        self.assertFalse((self.store / "runtimes").exists())

    def test_default_backends_follow_nvidia_detection(self):
        with patch.object(devices_module, "nvidia_gpu_present", return_value=False):
            code, out, _ = self.run_cli("--dry-run", "--json")
        data = json.loads(out)
        self.assertEqual(data["backends"], ["vulkan", "cpu"])
        self.assertTrue(any("cuda: not selected" in line for line in data["plan"]))
        with patch.object(devices_module, "nvidia_gpu_present", return_value=True):
            _, out, _ = self.run_cli("--dry-run", "--json")
        self.assertEqual(json.loads(out)["backends"], ["cuda", "vulkan", "cpu"])

    def test_backends_option_is_validated(self):
        code, out, _ = self.run_cli("--dry-run", "--json", "--backends", "vulkan,rocm")
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["code"], "usage")

    def test_setup_installs_selected_runtimes_and_skips_verified_ones(self):
        with patch("audio_transcript.adapters.download.download_verified", self.fetch):
            code, out, _ = self.run_cli("--yes", "--json", "--backends", "vulkan,cpu")
            self.assertEqual(code, 0, out)
            data = json.loads(out)
            self.assertEqual(len(data["installed_runtimes"]), 2)
            for name in ("llama.cpp-vulkan-b1", "llama.cpp-cpu-b1"):
                folder = self.store / "runtimes" / name
                self.assertTrue((folder / "llama-server.exe").is_file())
                self.assertTrue((folder / "runtime-manifest.json").is_file())
            self.assertFalse((self.store / "runtimes" / "llama.cpp-cuda").exists())
            again, out, _ = self.run_cli("--yes", "--json", "--backends", "vulkan,cpu")
        self.assertEqual(again, 0)
        self.assertEqual(json.loads(out)["installed_runtimes"], [])

    def test_tampered_download_fails_setup(self):
        def tampered(url, destination, *, sha256, size_bytes, progress=None):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"corrupt")

        with patch("audio_transcript.adapters.download.download_verified", tampered):
            code, out, _ = self.run_cli("--yes", "--json", "--backends", "cpu")
        self.assertEqual(code, 1)
        data = json.loads(out)
        self.assertEqual(data["error"]["code"], "download_failed")
        self.assertEqual(data["failed"], ["rt-cpu"])
        self.assertFalse((self.store / "runtimes" / "llama.cpp-cpu-b1").exists())

    def test_init_config_writes_the_runtimes_table(self):
        manifest = res.load_manifest(ROOT / "resources.json")
        profile = res.get_profile(manifest, None)
        template = (ROOT / "config.example.toml").read_text(encoding="utf-8")
        text = res.render_config(template, manifest, profile, self.store)
        self.assertIn('vulkan = "runtimes/llama.cpp-vulkan-b11389"', text)
        self.assertIn('cpu = "runtimes/llama.cpp-cpu-b11389"', text)
        probe = self.tmp / "probe.toml"
        probe.write_text(text, encoding="utf-8")
        self.assertEqual(set(load_config(probe).server.runtimes), {"cuda", "vulkan", "cpu"})


class ServerProfileAndDoctorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.store = self.tmp / "store"
        for backend in ("cuda", "vulkan", "cpu"):
            make_runtime(self.store / DEFAULT_RUNTIME_FOLDERS[backend], backend)
        text = SERVER_TABLES.format(store=self.store.as_posix()).replace(
            "[server]\n", '[server]\nruntime_dir = "runtimes/llama.cpp-cuda"\n', 1
        )
        self.config_path = self.tmp / "config.toml"
        self.config_path.write_text(text, encoding="utf-8")

    def run_cli(self, *argv, listings=None):
        probe = FakeProbe(listings or {})
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(devices_module, "list_devices", probe),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_server_profile_includes_the_selection(self):
        code, out, _ = self.run_cli(
            "server-profile",
            "--config",
            str(self.config_path),
            listings={"llama.cpp-cuda": CUDA_LISTING},
        )
        data = json.loads(out)
        self.assertEqual((code, data["backend"], data["device"]), (0, "cuda", "CUDA0"))
        self.assertFalse(data["selection"]["fallback_used"])
        self.assertTrue(data["runtime_dir"].endswith("llama.cpp-cuda"))
        code, out, _ = self.run_cli(
            "server-profile",
            "--config",
            str(self.config_path),
            listings={"llama.cpp-vulkan-b11389": VULKAN_LISTING},
        )
        data = json.loads(out)
        self.assertEqual((data["backend"], data["device"]), ("vulkan", "Vulkan0"))
        self.assertTrue(data["selection"]["fallback_used"])
        self.assertTrue(data["runtime_dir"].endswith("llama.cpp-vulkan-b11389"))
        self.assertIn("skipped cuda", "\n".join(data["selection"]["warning_lines"]))
        code, out, _ = self.run_cli("server-profile", "--config", str(self.config_path))
        data = json.loads(out)
        self.assertEqual((data["backend"], data["device"]), ("cpu", "none"))
        self.assertGreaterEqual(data["threads"], 1)

    def test_fixed_unusable_backend_fails_server_profile_clearly(self):
        path = self.tmp / "fixed.toml"
        path.write_text(
            self.config_path.read_text(encoding="utf-8").replace(
                "port = 8088", 'port = 8088\nbackend = "cuda"'
            ),
            encoding="utf-8",
        )
        code, out, _ = self.run_cli("server-profile", "--config", str(path), "--json")
        self.assertEqual(code, 1)
        error = json.loads(out)["error"]
        self.assertEqual(error["code"], "backend_unavailable")
        self.assertIn("never falls back", error["message"])

    def test_doctor_json_and_text_show_selection_and_warning(self):
        listings = {"llama.cpp-vulkan-b11389": VULKAN_LISTING}
        with (
            patch.object(
                cli,
                "_make_backend",
                return_value=SimpleNamespace(check=lambda: None, close=lambda: None),
            ),
            patch.object(cli, "_make_detector", return_value=None),
        ):
            code, out, _ = self.run_cli(
                "doctor", "--config", str(self.config_path), "--json", listings=listings
            )
            data = json.loads(out)
            check = next(c for c in data["checks"] if c["name"] == "inference_backend")
            self.assertTrue(check["ok"])
            self.assertEqual(check["selection"]["backend"], "vulkan")
            self.assertTrue(check["selection"]["fallback_used"])
            self.assertEqual(len(data["warnings"]), 1)
            self.assertIn("vulkan", data["warnings"][0])
            code, out, _ = self.run_cli(
                "doctor", "--config", str(self.config_path), listings=listings
            )
        self.assertEqual(code, 0)
        self.assertIn("Inference backend: vulkan Vulkan0", out)
        self.assertIn("WARNING: Inference backend fallback", out)


class RuntimeProvenanceTests(unittest.TestCase):
    INFO = {
        "backend": "vulkan",
        "device": "Vulkan0",
        "device_name": "NVIDIA GeForce RTX 5070 Ti",
        "fallback_used": True,
        "requested": "auto",
        "threads": None,
        "runtime_tag": "b11389",
        "skipped": [{"backend": "cuda", "reason": "no CUDA device reported"}],
    }

    def test_pipeline_records_runtime_in_reports_and_run_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "a.wav"
            source.parent.mkdir()
            source.write_bytes(b"original")
            backend = FakeBackend()
            backend.name = "llamacpp"
            config = make_config(
                root, backend="llamacpp", model="qwen2-audio-7b", archive_inputs=False
            )
            lines: list[str] = []
            pipeline = TranscriptionPipeline(
                config,
                FakeProcessor(chunk_count=1),
                backend,
                FileExporter(),
                status=lines.append,
                runtime=self.INFO,
            )
            result = pipeline.run()[0]
            out = root / "output" / result["job_id"]
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["inference_configuration"]["runtime"]["backend"], "vulkan")
            self.assertTrue(report["inference_configuration"]["runtime"]["fallback_used"])
            transcript = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(transcript["inference_configuration"]["runtime"]["device"], "Vulkan0")
            metadata = json.loads(
                (root / "process" / result["job_id"] / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["inference_configuration"]["runtime"]["backend"], "vulkan")
            self.assertIn(
                "Inference backend: vulkan Vulkan0 NVIDIA GeForce RTX 5070 Ti (fallback from cuda",
                (out / "run-report.md").read_text(encoding="utf-8"),
            )
            source_json = json.loads(
                (root / "output" / result["job_id"].split("/")[0] / "source.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(source_json["versions"][0]["runtime"]["backend"], "vulkan")
            plain = [line for line in lines if line.startswith("Backend:")]
            self.assertEqual(len(plain), 1)
            self.assertIn("(fallback from cuda: no CUDA device reported)", plain[0])

    def test_unmanaged_server_leaves_reports_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "a.wav"
            source.parent.mkdir()
            source.write_bytes(b"original")
            backend = FakeBackend()
            backend.name = "llamacpp"
            config = make_config(
                root, backend="llamacpp", model="qwen2-audio-7b", archive_inputs=False
            )
            lines: list[str] = []
            pipeline = TranscriptionPipeline(
                config, FakeProcessor(chunk_count=1), backend, FileExporter(), status=lines.append
            )
            result = pipeline.run()[0]
            report = json.loads(
                (root / "output" / result["job_id"] / "report.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("runtime", report["inference_configuration"])
            self.assertFalse([line for line in lines if line.startswith("Backend:")])

    def test_run_started_event_status_and_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            session = RunSession(config, command="run", runtime=self.INFO, status_out=None)
            seen = []
            session.bus.add_sink(CallbackSink(seen.append))
            session.start()
            started = next(e for e in seen if e.type == "run.started")
            self.assertEqual(started.data["runtime"]["backend"], "vulkan")
            self.assertTrue(started.data["runtime"]["fallback_used"])
            self.assertEqual(session.state.snapshot()["runtime"]["device"], "Vulkan0")
            session.finish(0)
            session.close()

    def test_live_header_shows_device_and_highlights_fallback(self):
        def state_for(info):
            seen = []
            bus = EventBus("r1", [CallbackSink(seen.append)])
            bus.emit("run.started", {"command": "run", "model": "m", "runtime": info})
            state = RunState()
            for event in seen:
                state.apply(event)
            return state

        normal = dict(self.INFO, backend="cuda", device="CUDA0", fallback_used=False, skipped=[])
        header = render(state_for(normal), 100, color=True, unicode=True)[0]
        self.assertIn("CUDA0 · RTX 5070 Ti", header)
        self.assertNotIn("(fallback)", header)
        self.assertNotIn("\x1b[1;33m", header)
        cpu = dict(self.INFO, backend="cpu", device="none", device_name="CPU", threads=12)
        header = render(state_for(cpu), 100, color=True, unicode=True)[0]
        self.assertIn("CPU (fallback)", header)
        self.assertIn("\x1b[1;33m", header)  # bold yellow
        plain_header = render(state_for(cpu), 100, color=False, unicode=False)[0]
        self.assertIn("CPU (fallback)", plain_header)
        self.assertNotIn("\x1b", plain_header)
        none = render(state_for(None), 100, color=False, unicode=True)[0]
        self.assertNotIn("fallback", none)


if __name__ == "__main__":
    unittest.main()

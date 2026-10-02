from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import tempfile
import tomllib
import unittest
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from audio_transcript.adapters import download
from audio_transcript.application import resources as res
from audio_transcript.cli import main
from audio_transcript.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = bytes(range(256)) * 40  # 10240 bytes
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.invalid/data/file.bin"


def make_resource(**overrides) -> res.Resource:
    values = {
        "id": "demo",
        "kind": "model",
        "platforms": ("any",),
        "url": URL,
        "revision": "rev1",
        "sha256": PAYLOAD_SHA,
        "size_bytes": len(PAYLOAD),
        "license": "MIT",
        "source": "https://example.invalid/page",
        "destination": "demo/file.bin",
    }
    values.update(overrides)
    return res.Resource(**values)


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self._offset = 0
        self.status = status

    def read(self, size: int = -1) -> bytes:
        end = len(self._body) if size < 0 else self._offset + size
        chunk = self._body[self._offset : end]
        self._offset += len(chunk)
        return chunk

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Stands in for urlopen: honours Range unless told not to."""

    def __init__(self, body: bytes, honour_range: bool = True):
        self.body = body
        self.honour_range = honour_range
        self.requests: list = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        header = request.get_header("Range")
        if header and self.honour_range:
            start = int(re.match(r"bytes=(\d+)-", header).group(1))
            return FakeResponse(self.body[start:], 206)
        return FakeResponse(self.body, 200)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "resources.json").read_text(encoding="utf-8"))

    def test_shipped_manifest_is_valid(self):
        manifest = res.parse_manifest(self.data)
        ids = [entry["id"] for entry in self.data["resources"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(manifest.default_profile, manifest.profiles)
        for profile in manifest.profiles.values():
            for rid in profile.resources:
                self.assertIn(rid, manifest.resources)
        for resource in manifest.resources.values():
            self.assertRegex(resource.sha256, r"^[0-9a-f]{64}$")
            self.assertFalse(PurePosixPath(resource.destination).is_absolute())
            self.assertNotIn("..", PurePosixPath(resource.destination).parts)
            self.assertTrue(resource.url.startswith("https://"))
            self.assertGreater(resource.size_bytes, 0)

    def test_model_urls_are_pinned_to_a_commit(self):
        for resource in res.parse_manifest(self.data).resources.values():
            if resource.kind in {"model", "projector"}:
                self.assertIn(resource.revision, resource.url)
                self.assertRegex(resource.revision, r"^[0-9a-f]{40}$")

    def test_invalid_manifests_are_rejected(self):
        def broken(mutate):
            data = json.loads(json.dumps(self.data))
            mutate(data)
            with self.assertRaises(ValueError):
                res.parse_manifest(data)

        broken(lambda d: d["resources"].append(dict(d["resources"][0])))
        broken(lambda d: d["resources"][0].update(sha256="ABC"))
        broken(lambda d: d["resources"][0].update(destination="/abs/file"))
        broken(lambda d: d["resources"][0].update(destination="C:/x/file"))
        broken(lambda d: d["resources"][0].update(destination="a/../../b"))
        broken(lambda d: d["resources"][0].update(kind="other"))
        broken(lambda d: d["profiles"]["qwen3-asr-1.7b-q8"]["resources"].append("missing"))
        broken(lambda d: d.update(schema_version=2))

    def test_silero_pins_match_the_installer(self):
        script = (ROOT / "scripts" / "install-silero-vad.ps1").read_text(encoding="utf-8")
        tag = re.search(r"\$tag = '([^']+)'", script).group(1)
        digest = re.search(r"\$expectedHash = '([0-9A-Fa-f]{64})'", script).group(1)
        resource = res.parse_manifest(self.data).resources["silero-vad-onnx"]
        self.assertEqual(resource.revision, tag)
        self.assertEqual(resource.sha256, digest.lower())
        self.assertIn(f"/{tag}/src/silero_vad/data/silero_vad.onnx", resource.url)
        self.assertEqual(resource.size_bytes, 2327524)

    def test_runtime_pins_match_the_installer(self):
        script = (ROOT / "scripts" / "install-llamacpp-cuda.ps1").read_text(encoding="utf-8")
        tag = re.search(r"\$Tag = '(b[0-9]+)'", script).group(1)
        cuda = re.search(r"\$CudaVersion = '([0-9.]+)'", script).group(1)
        names = {
            f"llama-{tag}-bin-win-cuda-{cuda}-x64.zip",
            f"cudart-llama-bin-win-cuda-{cuda}-x64.zip",
        }
        runtimes = [
            r for r in res.parse_manifest(self.data).resources.values() if r.kind == "runtime"
        ]
        self.assertEqual({r.url.rsplit("/", 1)[1] for r in runtimes}, names)
        for runtime in runtimes:
            self.assertEqual(runtime.revision, tag)
            self.assertIn(f"/releases/download/{tag}/", runtime.url)
            self.assertEqual(runtime.platforms, ("windows-x64",))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest = res.load_manifest(ROOT / "resources.json")
        self.profile = res.get_profile(self.manifest, "qwen3-asr-1.7b-q8")

    def test_plan_sizes_and_free_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            items = res.build_plan(
                self.manifest,
                self.profile,
                store,
                platform_id="windows-x64",
                hasher=download.sha256_file,
            )
            by_id = {item.resource.id: item for item in items}
            self.assertEqual(by_id["qwen3-asr-1.7b-q8-gguf"].status, "download")
            self.assertEqual(by_id["llamacpp-b11193-win-cuda-13.4-x64"].status, "installer")
            total = sum(item.download_bytes for item in items)
            self.assertEqual(total, 2165034944 + 641773984 + 2327524)
            lines = res.format_plan(items, store, 100)
            text = "\n".join(lines)
            self.assertIn("To download now: 2,809.1 MB", text)
            self.assertIn("WARNING: not enough free disk space", text)
            self.assertGreater(res.free_space(store), 0)

    def test_present_partial_conflicting_and_other_platform(self):
        manifest = res.parse_manifest(
            {
                "schema_version": 1,
                "resources": [
                    {
                        "id": "a",
                        "kind": "model",
                        "platforms": ["any"],
                        "url": URL,
                        "sha256": PAYLOAD_SHA,
                        "size_bytes": len(PAYLOAD),
                        "license": "MIT",
                        "source": "https://example.invalid",
                        "destination": "d/a.bin",
                    },
                    {
                        "id": "b",
                        "kind": "model",
                        "platforms": ["any"],
                        "url": URL,
                        "sha256": PAYLOAD_SHA,
                        "size_bytes": len(PAYLOAD),
                        "license": "MIT",
                        "source": "https://example.invalid",
                        "destination": "d/b.bin",
                    },
                    {
                        "id": "c",
                        "kind": "model",
                        "platforms": ["any"],
                        "url": URL,
                        "sha256": PAYLOAD_SHA,
                        "size_bytes": len(PAYLOAD),
                        "license": "MIT",
                        "source": "https://example.invalid",
                        "destination": "d/c.bin",
                    },
                    {
                        "id": "w",
                        "kind": "runtime",
                        "platforms": ["windows-x64"],
                        "url": URL,
                        "sha256": PAYLOAD_SHA,
                        "size_bytes": 5,
                        "license": "MIT",
                        "source": "https://example.invalid",
                        "destination": "rt",
                    },
                ],
                "profiles": {"p": {"resources": ["a", "b", "c", "w"]}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "d").mkdir()
            (store / "d" / "a.bin").write_bytes(PAYLOAD)
            (store / "d" / "b.bin").write_bytes(b"different")
            (store / "d" / "c.bin.part").write_bytes(PAYLOAD[:100])
            items = res.build_plan(
                manifest,
                manifest.profiles["p"],
                store,
                platform_id="linux-x64",
                hasher=download.sha256_file,
            )
            self.assertEqual(
                [item.status for item in items],
                ["verified", "conflict", "resume", "other-platform"],
            )
            self.assertEqual(items[2].download_bytes, len(PAYLOAD) - 100)


class DownloadTests(unittest.TestCase):
    def run_download(self, dest: Path, server, **overrides):
        args = {"sha256": PAYLOAD_SHA, "size_bytes": len(PAYLOAD)}
        args.update(overrides)
        with patch("audio_transcript.adapters.download.urllib.request.urlopen", server):
            download.download_verified(URL, dest, **args)

    def test_full_download_is_verified_and_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sub" / "file.bin"
            seen = []
            server = FakeServer(PAYLOAD)
            self.run_download(dest, server, progress=lambda *a: seen.append(a))
            self.assertEqual(dest.read_bytes(), PAYLOAD)
            self.assertFalse(dest.with_name("file.bin.part").exists())
            self.assertIsNone(server.requests[0].get_header("Range"))
            self.assertEqual(seen[-1][:2], (len(PAYLOAD), len(PAYLOAD)))

    def test_resume_uses_a_range_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "file.bin"
            dest.with_name("file.bin.part").write_bytes(PAYLOAD[:4000])
            server = FakeServer(PAYLOAD)
            self.run_download(dest, server)
            self.assertEqual(server.requests[0].get_header("Range"), "bytes=4000-")
            self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_server_ignoring_range_restarts_from_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "file.bin"
            dest.with_name("file.bin.part").write_bytes(PAYLOAD[:4000])
            self.run_download(dest, FakeServer(PAYLOAD, honour_range=False))
            self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_hash_mismatch_rejects_the_file_and_removes_the_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "file.bin"
            tampered = bytes([PAYLOAD[0] ^ 1]) + PAYLOAD[1:]
            with self.assertRaisesRegex(download.DownloadError, "SHA-256 mismatch"):
                self.run_download(dest, FakeServer(tampered))
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name("file.bin.part").exists())

    def test_short_download_keeps_the_part_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "file.bin"
            with self.assertRaisesRegex(download.DownloadError, "Incomplete"):
                self.run_download(dest, FakeServer(PAYLOAD[:5000]))
            self.assertFalse(dest.exists())
            self.assertEqual(dest.with_name("file.bin.part").stat().st_size, 5000)

    def test_existing_different_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "file.bin"
            dest.write_bytes(b"mine")
            server = FakeServer(PAYLOAD)
            with self.assertRaisesRegex(download.DownloadError, "Refusing to overwrite"):
                self.run_download(dest, server)
            self.assertEqual(dest.read_bytes(), b"mine")
            self.assertEqual(server.requests, [])


class ProvenanceTests(unittest.TestCase):
    def test_provenance_records_each_file_once(self):
        stamp = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "one.bin"
            second = Path(tmp) / "two.bin"
            res.write_provenance(first, make_resource(id="one"), now=stamp)
            res.write_provenance(second, make_resource(id="two"), now=stamp)
            res.write_provenance(first, make_resource(id="one", revision="rev2"), now=stamp)
            text = (Path(tmp) / "PROVENANCE.txt").read_text(encoding="utf-8")
            self.assertEqual(text.count("File:     one.bin"), 1)
            self.assertEqual(text.count("File:     two.bin"), 1)
            self.assertIn("Revision: rev2", text)
            for needle in (
                f"Source:   {URL}",
                f"SHA-256:  {PAYLOAD_SHA}",
                "License:  MIT",
                "Fetched:  2026-10-02T12:00:00Z",
            ):
                self.assertIn(needle, text)


class SetupCommandTests(unittest.TestCase):
    def manifest_file(self, tmp: Path) -> Path:
        path = tmp / "resources.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "resources": [
                        {
                            "id": "demo",
                            "kind": "model",
                            "platforms": ["any"],
                            "url": URL,
                            "revision": "rev1",
                            "sha256": PAYLOAD_SHA,
                            "size_bytes": len(PAYLOAD),
                            "license": "MIT",
                            "source": "https://example.invalid/page",
                            "destination": "demo/file.bin",
                        }
                    ],
                    "profiles": {"p": {"resources": ["demo"]}},
                }
            ),
            encoding="utf-8",
        )
        return path

    def run_main(self, argv, server=None, stdin=""):
        out, err = io.StringIO(), io.StringIO()
        patches = [patch("sys.stdin", io.StringIO(stdin))]
        if server is not None:
            patches.append(
                patch("audio_transcript.adapters.download.urllib.request.urlopen", server)
            )
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_prints_the_plan_and_downloads_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest_file(Path(tmp))
            server = FakeServer(PAYLOAD)
            code, out, _ = self.run_main(
                [
                    "setup",
                    "--manifest",
                    str(manifest),
                    "--profile",
                    "p",
                    "--store",
                    str(Path(tmp) / "store"),
                    "--dry-run",
                ],
                server,
            )
            self.assertEqual(code, 0)
            self.assertIn("to download", out)
            self.assertIn("Free disk space", out)
            self.assertEqual(server.requests, [])
            self.assertFalse((Path(tmp) / "store" / "demo").exists())

    def test_confirmation_declined_downloads_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest_file(Path(tmp))
            server = FakeServer(PAYLOAD)
            code, out, _ = self.run_main(
                ["setup", "--manifest", str(manifest), "--profile", "p", "--store", tmp + "/s"],
                server,
                stdin="n\n",
            )
            self.assertEqual(code, 1)
            self.assertEqual(server.requests, [])
            self.assertIn("Cancelled", out)

    def test_yes_downloads_verifies_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest_file(Path(tmp))
            store = Path(tmp) / "store"
            code, _, _ = self.run_main(
                ["setup", "--manifest", str(manifest), "--profile", "p", "--store", str(store)]
                + ["--yes"],
                FakeServer(PAYLOAD),
            )
            self.assertEqual(code, 0)
            self.assertEqual((store / "demo" / "file.bin").read_bytes(), PAYLOAD)
            self.assertIn(PAYLOAD_SHA, (store / "demo" / "PROVENANCE.txt").read_text("utf-8"))
            # A second run finds everything verified and does not contact the server.
            server = FakeServer(PAYLOAD)
            code, out, _ = self.run_main(
                ["setup", "--manifest", str(manifest), "--profile", "p", "--store", str(store)]
                + ["--yes"],
                server,
            )
            self.assertEqual(code, 0)
            self.assertIn("Nothing to download", out)
            self.assertEqual(server.requests, [])

    def test_conflicting_file_fails_without_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest_file(Path(tmp))
            store = Path(tmp) / "store"
            (store / "demo").mkdir(parents=True)
            (store / "demo" / "file.bin").write_bytes(b"other")
            code, _, err = self.run_main(
                ["setup", "--manifest", str(manifest), "--profile", "p", "--store", str(store)]
                + ["--yes"],
                FakeServer(PAYLOAD),
            )
            self.assertEqual(code, 1)
            self.assertIn("never overwritten", err)
            self.assertEqual((store / "demo" / "file.bin").read_bytes(), b"other")

    def test_missing_store_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest_file(Path(tmp))
            code, _, err = self.run_main(["setup", "--manifest", str(manifest), "--dry-run"])
            self.assertEqual(code, 2)
            self.assertIn("No resource store", err)


class InitConfigTests(unittest.TestCase):
    def run_init(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["init-config", *extra])
        return code, out.getvalue(), err.getvalue()

    def test_generated_config_parses_and_validates_for_each_profile(self):
        manifest = res.load_manifest(ROOT / "resources.json")
        for name in manifest.profiles:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "config.local.toml"
                store = Path(tmp) / "models"
                code, _, err = self.run_init(
                    "--store", str(store), "--profile", name, "--output", str(output)
                )
                self.assertEqual(code, 0, err)
                tomllib.loads(output.read_text(encoding="utf-8"))
                config = load_config(output)
                self.assertEqual(config.model, name)
                self.assertEqual(config.server.alias, name)
                self.assertEqual(config.resources.store, store.resolve())
                self.assertEqual(config.server.model_path.parent.parent, store.resolve())
                self.assertEqual(
                    config.vad_model_path, store.resolve() / "silero-vad" / "silero_vad.onnx"
                )

    def test_existing_file_is_kept_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "config.local.toml"
            output.write_text("keep me", encoding="utf-8")
            code, _, err = self.run_init("--store", tmp, "--output", str(output))
            self.assertEqual(code, 1)
            self.assertIn("--force", err)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")
            code, _, _ = self.run_init("--store", tmp, "--output", str(output), "--force")
            self.assertEqual(code, 0)
            self.assertIn("[resources]", output.read_text(encoding="utf-8"))

    def test_unknown_profile_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = self.run_init("--store", tmp, "--profile", "nope")
            self.assertEqual(code, 2)
            self.assertIn("Unknown profile", err)


if __name__ == "__main__":
    unittest.main()

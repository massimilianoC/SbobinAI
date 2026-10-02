"""Per-source, per-version layout, catalog, migration and archive tolerance (synthetic data)."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from audio_transcript.adapters.exporters import FileExporter
from audio_transcript.application import catalog as catalog_module
from audio_transcript.application import layout
from audio_transcript.application.migration import apply_migration, format_plan, plan_migration
from audio_transcript.application.pipeline import TranscriptionPipeline, _archive_duration_warning
from audio_transcript.cli import main
from audio_transcript.domain.models import DegenerateOutputError, Segment

try:
    from .fakes import FakeBackend, FakeProcessor, make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import FakeBackend, FakeProcessor, make_config


def _pipeline(config, processor=None, backend=None, lines=None):
    return TranscriptionPipeline(
        config,
        processor or FakeProcessor(chunk_count=2),
        backend or FakeBackend(),
        FileExporter(),
        status=(lines.append if lines is not None else (lambda _: None)),
    )


def _real_backend():
    backend = FakeBackend()
    backend.name = "llamacpp"
    return backend


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        (self.root / "input").mkdir()

    def media(self, name="Sample Talk.wav", data=b"synthetic media bytes"):
        path = self.root / "input" / name
        path.write_bytes(data)
        return path


class SourceKeyTests(WorkspaceCase):
    def test_key_is_stable_across_rename_move_and_mtime(self):
        first = self.media("one.wav", b"same bytes")
        key = layout.source_key(first)
        self.assertEqual(len(key), layout.SOURCE_KEY_LENGTH)
        moved = self.root / "elsewhere.mp3"
        os.rename(first, moved)
        os.utime(moved, ns=(1_000_000_000, 1_000_000_000))
        self.assertEqual(layout.source_key(moved), key)
        moved.write_bytes(b"other bytes")
        self.assertNotEqual(layout.source_key(moved), key)

    def test_key_reads_only_head_and_tail_blocks(self):
        block = layout.SOURCE_KEY_BLOCK
        path = self.root / "big.bin"
        path.write_bytes(b"a" * block + b"m" * block + b"z" * block)
        key = layout.source_key(path)
        data = bytearray(path.read_bytes())
        data[block + 10] = ord("X")  # middle: not part of the identity by design
        path.write_bytes(bytes(data))
        self.assertEqual(layout.source_key(path), key)
        data[-1] = ord("Y")  # tail changes the key
        path.write_bytes(bytes(data))
        self.assertNotEqual(layout.source_key(path), key)
        data[-1] = ord("z")
        data = data[:-1]  # size changes the key
        path.write_bytes(bytes(data))
        self.assertNotEqual(layout.source_key(path), key)

    def test_source_folder_reuses_sanitising_rule(self):
        self.assertEqual(
            layout.source_folder_name("café audio.wav", "0123456789ab"), "caf_audio-0123456789ab"
        )
        self.assertEqual(len(layout.safe_stem("x" * 100 + ".wav")), 64)


class VersionNamingTests(WorkspaceCase):
    def test_version_folder_name_parts(self):
        from datetime import UTC, datetime

        created = datetime(2026, 10, 2, 11, 5, 59, tzinfo=UTC)
        fp = "ab12cd34" + "0" * 56
        self.assertEqual(
            layout.version_folder_name(created, "Qwen3-ASR 1.7B Q8_0", None, fp),
            "20261002T1105Z_qwen3-asr-1.7b-q8-0_full_ab12cd34",
        )
        self.assertEqual(
            layout.version_folder_name(created, "m", 600.0, fp),
            "20261002T1105Z_m_first600s_ab12cd34",
        )
        self.assertEqual(
            layout.version_folder_name(created, "m", 12.5, fp),
            "20261002T1105Z_m_first12p5s_ab12cd34",
        )
        parsed = layout.parse_version_folder("20261002T1105Z_m_first600s_ab12cd34")
        self.assertEqual(parsed["fingerprint_prefix"], "ab12cd34")
        self.assertIsNone(layout.parse_version_folder("notes"))

    def test_lookup_finds_by_fingerprint_suffix_and_rejects_duplicates(self):
        process, output = self.root / "process", self.root / "output"
        fp = "feedbeef" + "1" * 56
        (process / "s-aaaaaaaaaaaa" / "20260101T0000Z_m_full_feedbeef").mkdir(parents=True)
        (process / "s-aaaaaaaaaaaa" / "20260101T0000Z_m_full_00000000").mkdir()
        (output / "s-aaaaaaaaaaaa" / "20260101T0000Z_m_full_feedbeef").mkdir(parents=True)
        self.assertEqual(
            layout.find_version_folder([process, output], "s-aaaaaaaaaaaa", fp),
            "20260101T0000Z_m_full_feedbeef",
        )
        self.assertIsNone(layout.find_version_folder([process, output], "s-aaaaaaaaaaaa", "9" * 64))
        (output / "s-aaaaaaaaaaaa" / "20260202T0000Z_other_full_feedbeef").mkdir()
        with self.assertRaisesRegex(layout.LayoutError, "More than one"):
            layout.find_version_folder([process, output], "s-aaaaaaaaaaaa", fp)

    def test_duplicate_versions_fail_the_job_without_touching_data(self):
        source = self.media()
        config = make_config(self.root)
        pipeline = _pipeline(config)
        job_id = pipeline.run([source])[0]["job_id"]
        folder, version = job_id.split("/")
        duplicate = version.replace("20", "21", 1)
        (self.root / "process" / folder / duplicate).mkdir()
        result = pipeline.run([source])[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("More than one", result["error"])
        self.assertTrue((self.root / "process" / folder / version / "metadata.json").is_file())


class PipelineLayoutTests(WorkspaceCase):
    def test_tree_rename_skip_and_resume_lookup(self):
        source = self.media("Sample Talk.wav")
        config = make_config(self.root)
        processor = FakeProcessor(chunk_count=2)
        result = _pipeline(config, processor).run([source])[0]
        folder, version = result["job_id"].split("/")
        key = layout.source_key(source)
        self.assertEqual(folder, f"Sample_Talk-{key}")
        for base in ("process", "output"):
            self.assertTrue((self.root / base / folder / version).is_dir())
        self.assertTrue((self.root / "process" / folder / version / "metadata.json").is_file())
        self.assertTrue((self.root / "output" / folder / version / "transcript.json").is_file())
        self.assertTrue((self.root / "output" / folder / "source.json").is_file())
        self.assertTrue((self.root / "output" / "catalog.json").is_file())
        # Same bytes under another name: same source folder, completed version is skipped.
        renamed = source.with_name("renamed.wav")
        os.rename(source, renamed)
        os.utime(renamed, ns=(2_000_000_000, 2_000_000_000))
        again = _pipeline(config, processor).run([renamed])[0]
        self.assertEqual(again["status"], "skipped")
        self.assertEqual(again["job_id"], result["job_id"])
        self.assertEqual(processor.prepare_calls, 1)
        self.assertEqual(len([p for p in (self.root / "process").iterdir() if p.is_dir()]), 1)

    def test_source_json_and_catalog_content_and_ordering(self):
        source = self.media()
        _pipeline(make_config(self.root, model="model-a")).run([source])
        second = _pipeline(make_config(self.root, model="model-b", max_duration=1.0))
        result = second.run([source])[0]
        folder = result["job_id"].split("/")[0]
        info = _read(self.root / "output" / folder / "source.json")
        self.assertEqual(info["schema_version"], 1)
        self.assertEqual(info["source_key"], layout.source_key(source))
        self.assertEqual(info["original_names"], ["Sample Talk.wav"])
        self.assertEqual(info["size_bytes"], source.stat().st_size)
        self.assertEqual(info["location"], {"kind": "input", "path": "input/Sample Talk.wav"})
        self.assertEqual(info["version_count"], 2)
        newest, oldest = info["versions"]
        self.assertEqual(newest["model"], "model-b")
        self.assertEqual(oldest["model"], "model-a")
        self.assertEqual(newest["scope"]["kind"], "bounded")
        self.assertEqual(oldest["scope"]["kind"], "full")
        self.assertIn("_first1s_", newest["version"])
        self.assertEqual(oldest["status"], "completed")
        self.assertEqual(oldest["completion_percent"], 100.0)
        self.assertEqual(oldest["chunks"], {"total": 2, "ok": 2, "no_speech": 0, "failed": 0})
        self.assertEqual(oldest["backend"], "mock")
        self.assertIsNotNone(oldest["real_time_factor"])
        self.assertIsNone(oldest["confidence"])
        self.assertEqual(oldest["analysed_duration_seconds"], 2.0)
        self.assertEqual(oldest["artifacts"]["kind"], "final")
        self.assertTrue(
            (self.root / "output" / folder / oldest["artifacts"]["transcript"]).is_file()
        )
        self.assertTrue(
            (self.root / "output" / folder / oldest["artifacts"]["run_report"]).is_file()
        )
        catalog = _read(self.root / "output" / "catalog.json")
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(catalog["source_count"], 1)
        entry = catalog["sources"][0]
        self.assertEqual(entry["folder"], folder)
        self.assertEqual(entry["version_count"], 2)
        self.assertEqual(entry["latest_version"]["model"], "model-b")
        self.assertEqual(entry["location"], "input")
        for path in (
            self.root / "output" / folder / "source.json",
            self.root / "output" / "catalog.json",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(Path(self._temporary.name).name, text)  # no absolute machine paths

    def test_catalog_orders_sources_by_latest_activity(self):
        first = self.media("first.wav", b"first")
        second = self.media("second.wav", b"second")
        config = make_config(self.root)
        _pipeline(config).run([first])
        _pipeline(config).run([second])
        names = [e["original_name"] for e in _read(self.root / "output/catalog.json")["sources"]]
        self.assertEqual(names, ["second.wav", "first.wav"])
        _pipeline(make_config(self.root, model="later")).run([first])
        names = [e["original_name"] for e in _read(self.root / "output/catalog.json")["sources"]]
        self.assertEqual(names, ["first.wav", "second.wav"])

    def test_location_archived_external_and_missing(self):
        source = self.media()
        external = self.root / "external.wav"
        external.write_bytes(b"external bytes")
        config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b")
        results = _pipeline(config, backend=_real_backend()).run([source, external])
        archived_folder = results[0]["job_id"].split("/")[0]
        external_folder = results[1]["job_id"].split("/")[0]
        info = _read(self.root / "output" / archived_folder / "source.json")
        self.assertEqual(
            info["location"],
            {"kind": "processed", "path": f"processed/{archived_folder}/Sample Talk.wav"},
        )
        self.assertEqual(
            _read(self.root / "output" / external_folder / "source.json")["location"],
            {"kind": "external", "path": None},
        )
        external.unlink()
        catalog_module.rebuild_all(config)
        self.assertEqual(
            _read(self.root / "output" / external_folder / "source.json")["location"]["kind"],
            "missing",
        )

    def test_incomplete_then_resume_records_runs_and_run_reports(self):
        source = self.media()

        def flaky(chunk, temperature):
            if chunk.index == 1:
                raise DegenerateOutputError("loop")
            return [Segment(chunk.start, chunk.end, "secret words")]

        config = make_config(self.root, fallback_temperatures=(), split_on_failure=False)
        lines = []
        first = _pipeline(config, backend=FakeBackend(behavior=flaky), lines=lines).run([source])[0]
        self.assertEqual(first["status"], "incomplete")
        folder, version = first["job_id"].split("/")
        output = self.root / "output" / folder / version
        self.assertTrue((output / "intermediate" / "run-report.md").is_file())
        self.assertFalse((output / "run-report.md").exists())
        info = _read(self.root / "output" / folder / "source.json")
        self.assertEqual(info["versions"][0]["status"], "incomplete")
        self.assertEqual(info["versions"][0]["completion_percent"], 50.0)
        self.assertEqual(info["versions"][0]["chunks"]["failed"], 1)
        self.assertTrue(lines[-1].startswith("Timings: total "))

        lines.clear()
        done = _pipeline(config, lines=lines).run([source])[0]
        self.assertEqual(done["status"], "completed")
        metadata = _read(self.root / "process" / folder / version / "metadata.json")
        self.assertEqual([r["status"] for r in metadata["runs"]], ["incomplete", "completed"])
        self.assertEqual(metadata["runs"][1]["resumed_chunks"], 1)
        report = (output / "run-report.md").read_text(encoding="utf-8")
        self.assertIn("Cumulative wall time over 2 run(s)", report)
        self.assertIn("Completion: 100%", report)
        self.assertIn("not separable", report)
        self.assertNotIn("secret words", report)
        stages = _read(output / "report.json")["execution"]["stages_seconds"]
        for key in ("probe", "preparation", "inference", "export"):
            self.assertIn(key, stages)
        self.assertIn("| export ", lines[-1])
        self.assertEqual(
            _read(self.root / "output" / folder / "source.json")["versions"][0]["status"],
            "completed",
        )

    def test_prepare_only_version_is_listed(self):
        source = self.media()
        config = make_config(self.root, prepare_only=True)
        result = TranscriptionPipeline(
            config, FakeProcessor(chunk_count=2), None, None, status=lambda _: None
        ).run([source])[0]
        self.assertEqual(result["status"], "prepared")
        info = _read(self.root / "output" / result["job_id"].split("/")[0] / "source.json")
        self.assertEqual(info["versions"][0]["status"], "prepared")
        self.assertEqual(info["versions"][0]["completion_percent"], 0.0)


class ArchiveToleranceTests(unittest.TestCase):
    @staticmethod
    def _metadata(source, audio):
        return {
            "source_metadata": {"duration_seconds": source},
            "audio_metadata": {"duration_seconds": audio, "sample_rate": 16000},
        }

    def test_ffprobe_rounding_does_not_block_a_full_run(self):
        self.assertIsNone(_archive_duration_warning(self._metadata(7545.963, 7545.9626875)))
        self.assertIsNone(_archive_duration_warning(self._metadata(7545.99, 7545.95)))

    def test_bounded_run_is_still_rejected(self):
        self.assertIn(
            "Only part of the source",
            _archive_duration_warning(self._metadata(7546.0, 600.0)),
        )
        self.assertIn(
            "Only part of the source",
            _archive_duration_warning(self._metadata(7546.0, 7545.5)),
        )


class MigrationTests(WorkspaceCase):
    OLD = "Sample_Talk-0123456789abcd"

    def _legacy(self, *, archived: bool, keep_media: bool = True):
        """Run in the new layout, then rewrite the result into the old flat layout."""
        source = self.media()
        if archived:
            config = make_config(self.root, backend="llamacpp", model="qwen2-audio-7b")
            backend = _real_backend()
        else:
            config = make_config(self.root)
            backend = FakeBackend()
        result = _pipeline(config, backend=backend).run([source])[0]
        folder, version = result["job_id"].split("/")
        base = self.root
        os.rename(base / "process" / folder / version, base / "process" / self.OLD)
        os.rename(base / "output" / folder / version, base / "output" / self.OLD)
        for name in ("source.json",):
            (base / "output" / folder / name).unlink()
        (base / "output" / folder).rmdir()
        (base / "process" / folder).rmdir()
        (base / "output" / "catalog.json").unlink()
        if archived:
            os.rename(base / "processed" / folder, base / "processed" / self.OLD)
            if not keep_media:
                (base / "processed" / self.OLD / source.name).unlink()
        metadata_path = base / "process" / self.OLD / "metadata.json"
        metadata = _read(metadata_path)
        size = source.stat().st_size if source.exists() else len(b"synthetic media bytes")
        for key in (
            "created_at", "runs", "source_folder", "version_folder", "source_key",
            "last_seen_path", "scope", "model", "language", "response_mode",
        ):  # fmt: skip
            metadata.pop(key, None)
        metadata["job_id"] = self.OLD
        metadata["identity"] = {"source": str(source), "size": size, "mtime_ns": 1}
        if archived:
            metadata["archived_source_path"] = str(base / "processed" / self.OLD / source.name)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (base / "process" / "stray-notes").mkdir()
        (base / "process" / "stray-notes" / "readme.txt").write_text("keep me", encoding="utf-8")
        return config, source, result, metadata

    def test_dry_run_changes_nothing_and_apply_moves_then_rerun_skips(self):
        config, source, _, metadata = self._legacy(archived=True)
        key = layout.source_key(config.processed_dir / self.OLD / source.name)
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        items = plan_migration(config)
        statuses = {item.old: item for item in items}
        self.assertEqual(statuses["stray-notes"].status, "skipped")
        self.assertIn("not a job", statuses["stray-notes"].reason)
        job = statuses[self.OLD]
        self.assertEqual(job.status, "move")
        self.assertEqual(job.source_folder, f"Sample_Talk-{key}")
        self.assertEqual(job.key_basis, "content")
        self.assertRegex(job.version_folder, r"^\d{8}T\d{4}Z_qwen2-audio-7b_full_[0-9a-f]{8}$")
        self.assertEqual(job.version_folder.split("_")[-1], metadata["config_fingerprint"][:8])
        table = "\n".join(format_plan(items))
        self.assertIn("skipped (not a job)", table)
        self.assertIn(job.version_folder, table)
        self.assertEqual(
            before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        )

        log_path, summary = apply_migration(config, items)
        self.assertEqual(summary["migrated"], 1)
        target_process = self.root / "process" / job.source_folder / job.version_folder
        self.assertTrue((target_process / "metadata.json").is_file())
        self.assertTrue(
            (
                self.root / "output" / job.source_folder / job.version_folder / "report.json"
            ).is_file()
        )
        self.assertTrue((self.root / "processed" / job.source_folder / source.name).is_file())
        self.assertFalse((self.root / "process" / self.OLD).exists())
        self.assertFalse((self.root / "processed" / self.OLD).exists())
        self.assertEqual(
            (self.root / "process" / "stray-notes" / "readme.txt").read_text(encoding="utf-8"),
            "keep me",
        )
        log = _read(log_path)
        self.assertEqual(
            sorted((m["root"], m["from"]) for m in log["moves"]),
            [("output", self.OLD), ("process", self.OLD), ("processed", self.OLD)],
        )
        self.assertTrue(all(m["result"] == "moved" for m in log["moves"]))
        info = _read(self.root / "output" / job.source_folder / "source.json")
        self.assertEqual(info["versions"][0]["status"], "completed")
        # Migrated versions expose the same scope shape as new ones.
        self.assertEqual(
            info["versions"][0]["scope"],
            {"kind": "full", "max_duration_seconds": None, "label": "full"},
        )
        self.assertEqual(info["location"]["kind"], "processed")
        self.assertEqual(_read(self.root / "output" / "catalog.json")["source_count"], 1)
        # Idempotent: nothing left to migrate.
        again = plan_migration(config)
        self.assertEqual(sorted(i.status for i in again), ["already-migrated", "skipped"])

        # A re-run with the same input and settings finds the migrated version and skips it.
        processor = FakeProcessor(chunk_count=2)
        archived = self.root / "processed" / job.source_folder / source.name
        rerun = _pipeline(config, processor, _real_backend()).run([archived])[0]
        self.assertEqual(rerun["status"], "skipped")
        self.assertEqual(processor.prepare_calls, 0)
        self.assertEqual(rerun["job_id"], f"{job.source_folder}/{job.version_folder}")

    def test_missing_media_falls_back_to_metadata_key(self):
        config, source, _, _ = self._legacy(archived=True, keep_media=False)
        items = plan_migration(config)
        job = next(i for i in items if i.old == self.OLD)
        self.assertEqual(job.key_basis, "metadata")
        self.assertIn("key derived", job.reason)
        apply_migration(config, items)
        info = _read(self.root / "output" / job.source_folder / "source.json")
        self.assertEqual(info["source_key_basis"], "metadata")
        self.assertEqual(info["location"]["kind"], "missing")

    def test_existing_target_is_never_overwritten(self):
        config, source, _, _ = self._legacy(archived=False)
        job = next(i for i in plan_migration(config) if i.old == self.OLD)
        blocker = self.root / "output" / job.source_folder / job.version_folder
        blocker.mkdir(parents=True)
        (blocker / "keep.txt").write_text("precious", encoding="utf-8")
        items = plan_migration(config)
        conflict = next(i for i in items if i.old == self.OLD)
        self.assertEqual(conflict.status, "conflict")
        self.assertIn("refusing to overwrite", conflict.reason)
        _, summary = apply_migration(config, items)
        self.assertEqual(summary["refused"], 1)
        self.assertEqual(summary["migrated"], 0)
        self.assertTrue((self.root / "process" / self.OLD / "metadata.json").is_file())
        self.assertEqual((blocker / "keep.txt").read_text(encoding="utf-8"), "precious")

    def test_apply_refuses_when_a_target_appears_after_planning(self):
        config, source, _, _ = self._legacy(archived=False)
        items = plan_migration(config)
        job = next(i for i in items if i.old == self.OLD)
        late = self.root / "output" / job.source_folder / job.version_folder
        late.mkdir(parents=True)
        log_path, summary = apply_migration(config, items)
        self.assertEqual(summary["failed"], 1)
        # The process move that had already happened is rolled back, never deleted.
        self.assertTrue((self.root / "process" / self.OLD / "metadata.json").is_file())
        self.assertTrue((self.root / "output" / self.OLD / "report.json").is_file())
        results = [m["result"] for m in _read(log_path)["moves"]]
        self.assertIn("rolled-back", results)

    def test_cli_catalog_and_migrate_layout(self):
        config, source, _, _ = self._legacy(archived=False)
        args = [
            "--input-dir", str(config.input_dir),
            "--process-dir", str(config.process_dir),
            "--processed-dir", str(config.processed_dir),
            "--output-dir", str(config.output_dir),
        ]  # fmt: skip
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["migrate-layout", *args]), 0)
        self.assertIn("Dry run", stdout.getvalue())
        self.assertIn("skipped (not a job)", stdout.getvalue())
        self.assertTrue((self.root / "process" / self.OLD).is_dir())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["migrate-layout", "--apply", *args]), 0)
        self.assertIn("Migrated 1 job(s)", stdout.getvalue())
        self.assertFalse((self.root / "process" / self.OLD).exists())
        catalog_path = self.root / "output" / "catalog.json"
        first = catalog_path.read_bytes()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["catalog", *args]), 0)
        self.assertIn("1 source(s), 1 version(s)", stdout.getvalue())
        self.assertEqual(catalog_path.read_bytes(), first)  # rebuild is idempotent


if __name__ == "__main__":
    unittest.main()

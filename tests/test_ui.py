"""Console renderer snapshots, UI mode resolution and cursor safety (no terminal needed)."""

import io
import re
import unittest

from audio_transcript.application.events import CallbackSink, EventBus
from audio_transcript.application.progress import RunState
from audio_transcript.ui import resolve_ui_mode
from audio_transcript.ui.live import Glyphs, LiveConsole, render, sparkline, supports_unicode

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def visible(line: str) -> str:
    return ANSI.sub("", line)


def synthetic_state(*, resources=True, failed=1, finished=False) -> RunState:
    """A mid-run state built only from events, as a real run would produce it."""
    seen = []
    ticks = iter(x * 1.7 for x in range(10_000))
    bus = EventBus("20261002T120000Z-ab12", [CallbackSink(seen.append)], clock=lambda: next(ticks))
    bus.emit(
        "run.started",
        {
            "command": "run",
            "model": "qwen3-asr-1.7b-q8",
            "backend": "llamacpp",
            "language": "it",
            "parallel_requests": 2,
            "queue_size": 3,
        },
    )
    job = "interview-1a2b3c4d5e6f/20261002T1200Z_qwen3-asr-1.7b-q8_full_ab12cd34"
    bus.emit(
        "job.started",
        {
            "source": "interview.m4a",
            "version": job.split("/")[1],
            "scope": "full",
            "index": 2,
            "total": 3,
        },
        job_id=job,
    )
    for stage, seconds in (("probe", 0.4), ("audio_extraction", 3.1), ("speech_detection", 5.2)):
        bus.emit("stage.finished", {"stage": stage, "seconds": seconds}, job_id=job)
    bus.emit(
        "stage.finished", {"stage": "chunk_slicing", "seconds": None, "reused": True}, job_id=job
    )
    bus.emit("stage.started", {"stage": "inference"}, job_id=job)
    bus.emit(
        "job.prepared",
        {"analysed_seconds": 850.0, "speech_seconds": 662.0, "chunk_count": 57},
        job_id=job,
    )
    bus.emit(
        "progress",
        {
            "done_chunks": 23,
            "total_chunks": 57,
            "percent": 40.4,
            "audio_done_s": 341.0,
            "audio_total_s": 850.0,
            "elapsed_s": 42.0,
            "eta_s": 104.0,
            "real_time_factor": 0.123,
            "throughput_x": 8.1,
            "tokens": 1240.0,
            "ok": 21,
            "no_speech": 2 - 0,
            "failed": failed,
            "fallbacks": 1,
            "splits": 0,
            "confidence_mean": 0.9312,
            "in_flight": [23, 24],
        },
        job_id=job,
    )
    bus.emit(
        "log",
        {"level": "info", "message": "Chunk 22/57 [310.0-325.0s, 15.0s audio] t=0 ok 4.1s 55 tok"},
        job_id=job,
    )
    bus.emit(
        "log",
        {"level": "info", "message": "Chunk 23/57 [325.0-338.5s, 13.5s audio] t=0 ok 3.8s 51 tok"},
        job_id=job,
    )
    bus.emit(
        "log",
        {
            "level": "error",
            "message": "Chunk 21/57 [295.0-310.0s, 15.0s audio] t=0.4 DegenerateOutputError(loop) 9.2s FAILED after ladder -> continuing",
        },
        job_id=job,
    )
    bus.emit("warning", {"code": "x", "message": "Resource monitor: nvidia-smi restarted"})
    if resources:
        for step in range(12):
            bus.emit(
                "resource.sample",
                {
                    "scope": "system",
                    "source": {"host": "windows", "gpu": "nvidia-smi"},
                    "cpu_percent": 20.0 + step * 3,
                    "ram_used_mb": 12000.0 + step * 50,
                    "ram_total_mb": 32768.0,
                    "process_rss_mb": 410.0,
                    "gpu_util_percent": 60.0 + step * 3,
                    "gpu_mem_used_mb": 4100.0,
                    "gpu_mem_total_mb": 16303.0,
                    "gpu_power_w": 180.0 + step,
                    "gpu_temp_c": 58.0 + step / 2,
                },
            )
    if finished:
        bus.emit("run.finished", {"exit_status": 0, "jobs": [], "wall_seconds": 252.0})
    state = RunState()
    for event in seen:
        state.apply(event)
    state.tick(252.0)
    return state


class RenderTests(unittest.TestCase):
    def test_lines_never_exceed_the_width_at_any_size(self):
        state = synthetic_state()
        for width in (60, 100, 140):
            for color in (True, False):
                for unicode in (True, False):
                    with self.subTest(width=width, color=color, unicode=unicode):
                        lines = render(state, width, color=color, unicode=unicode)
                        self.assertTrue(lines)
                        for line in lines:
                            self.assertLessEqual(len(visible(line)), width - 1)

    def test_snapshot_contains_every_section_at_100_columns(self):
        text = "\n".join(visible(line) for line in render(synthetic_state(), 100, color=False))
        for fragment in (
            "SbobinAI",
            "qwen3-asr-1.7b-q8",
            "parallel 2",
            "File 2/3",
            "interview.m4a",
            "probe 0.4s",
            "slice cached",
            "transcribe",
            "23/57",
            " 40%",
            "ETA 1m44s",
            "x8.1 realtime",
            "ok 21",
            "failed 1",
            "fallbacks 1",
            "in flight #24 #25",
            "tokens 1,240",
            "confidence 0.93",
            "CPU",
            "RAM",
            "GPU",
            "VRAM",
            "PWR",
            "TEMP",
            "12.3/32.0 GB",
            "DegenerateOutputError(loop)",
            "activity",
        ):
            self.assertIn(fragment, text, fragment)

    def test_narrow_width_truncates_instead_of_wrapping(self):
        lines = [visible(line) for line in render(synthetic_state(), 60, color=False)]
        self.assertTrue(any(line.endswith("…") for line in lines))
        self.assertTrue(all(len(line) <= 59 for line in lines))

    def test_missing_resources_show_n_a(self):
        text = "\n".join(
            visible(line) for line in render(synthetic_state(resources=False), 120, color=False)
        )
        self.assertIn("CPU", text)
        self.assertGreaterEqual(text.count("n/a"), 8)
        self.assertIn("n/a/n/a GB", text)

    def test_ascii_fallback_uses_only_ascii(self):
        lines = render(synthetic_state(), 120, color=False, unicode=False)
        for line in lines:
            self.assertTrue(line.isascii(), line)
        text = "\n".join(lines)
        self.assertIn("[", text)
        self.assertIn("#", text)
        self.assertIn("+ probe", text)

    def test_no_color_has_no_escape_codes_and_color_has(self):
        state = synthetic_state()
        self.assertFalse(any("\x1b" in line for line in render(state, 100, color=False)))
        colored = "\n".join(render(state, 100, color=True))
        self.assertIn("\x1b[", colored)
        self.assertIn("\x1b[31", colored)  # failures and errors are highlighted

    def test_empty_state_renders(self):
        lines = render(RunState(), 80, color=False)
        self.assertIn("Waiting for the first file", "\n".join(lines))

    def test_finished_state_header(self):
        header = visible(render(synthetic_state(finished=True), 100, color=False)[0])
        self.assertIn("done in 04:12", header)

    def test_sparkline_scaling(self):
        glyphs = Glyphs(True)
        self.assertEqual(sparkline([0, 100], glyphs, 2, 0, 100), "▁█")
        self.assertEqual(sparkline([], glyphs, 4), "    ")
        self.assertEqual(len(sparkline([5] * 30, glyphs, 6, 0, 10)), 6)

    def test_supports_unicode_follows_the_stream_encoding(self):
        class Stream:
            def __init__(self, encoding):
                self.encoding = encoding

        self.assertTrue(supports_unicode(Stream("utf-8")))
        self.assertFalse(supports_unicode(Stream("ascii")))
        self.assertFalse(supports_unicode(Stream("cp437")))


class ModeTests(unittest.TestCase):
    def test_auto_is_plain_when_not_a_tty(self):
        self.assertEqual(
            resolve_ui_mode("auto", isatty=False, environ={}, enable_vt=lambda: True), "plain"
        )

    def test_auto_is_plain_under_ci_or_dumb_terminal(self):
        for env in ({"CI": "true"}, {"TERM": "dumb"}):
            with self.subTest(env=env):
                self.assertEqual(
                    resolve_ui_mode("auto", isatty=True, environ=env, enable_vt=lambda: True),
                    "plain",
                )

    def test_auto_is_plain_when_vt_cannot_be_enabled(self):
        self.assertEqual(
            resolve_ui_mode("auto", isatty=True, environ={}, enable_vt=lambda: False), "plain"
        )

    def test_auto_is_live_on_a_capable_terminal(self):
        self.assertEqual(
            resolve_ui_mode("auto", isatty=True, environ={"TERM": "xterm"}, enable_vt=lambda: True),
            "live",
        )

    def test_explicit_modes(self):
        self.assertEqual(resolve_ui_mode("plain", isatty=True, environ={}), "plain")
        self.assertEqual(resolve_ui_mode("jsonl", isatty=True, environ={}), "jsonl")
        self.assertEqual(
            resolve_ui_mode("live", isatty=False, environ={"CI": "1"}, enable_vt=lambda: True),
            "live",
        )
        self.assertEqual(
            resolve_ui_mode("live", isatty=True, environ={}, enable_vt=lambda: False), "plain"
        )


class LiveConsoleTests(unittest.TestCase):
    def test_cursor_is_hidden_and_always_restored(self):
        stream = io.StringIO()
        console = LiveConsole(stream, color=False, unicode=True, width=lambda: 100, max_fps=50)
        bus = EventBus("r1", [console])
        bus.emit("run.started", {"command": "run", "model": "m"})
        bus.emit("log", {"level": "info", "message": "hello activity"})
        bus.close()
        out = stream.getvalue()
        self.assertTrue(out.startswith("\x1b[?25l"))
        self.assertTrue(out.rstrip().endswith("\x1b[?25h") or out.endswith("\x1b[?25h"))
        self.assertIn("hello activity", out)

    def test_cursor_restored_when_the_run_raises(self):
        stream = io.StringIO()
        console = LiveConsole(stream, color=False, unicode=True, width=lambda: 80)
        try:
            bus = EventBus("r1", [console])
            bus.emit("run.started", {"command": "run"})
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            bus.close()
        self.assertTrue(stream.getvalue().endswith("\x1b[?25h"))

    def test_redraw_moves_up_by_the_previous_frame_height(self):
        stream = io.StringIO()
        console = LiveConsole(stream, color=False, unicode=True, width=lambda: 100)
        console.handle(_event("run.started", {"command": "run", "model": "m"}))
        console.draw()
        height = console._height
        console.handle(_event("log", {"level": "info", "message": "again"}))
        console.draw()
        self.assertIn(f"\x1b[{height}F", stream.getvalue())
        console.close()

    def test_unencodable_stream_switches_to_ascii(self):
        class Strict(io.StringIO):
            def write(self, text):
                if not text.isascii() and self.strict:
                    raise UnicodeEncodeError("ascii", text, 0, 1, "no")
                return super().write(text)

        stream = Strict()
        stream.strict = True
        console = LiveConsole(stream, color=False, unicode=True, width=lambda: 100)
        console.handle(_event("run.started", {"command": "run", "model": "m"}))
        console.draw()  # fails once, then falls back
        console._dirty = True
        console.draw()
        self.assertFalse(console.unicode)
        console.close()


def _event(kind, data):
    seen = []
    EventBus("r1", [CallbackSink(seen.append)]).emit(kind, data)
    return seen[0]


if __name__ == "__main__":
    unittest.main()

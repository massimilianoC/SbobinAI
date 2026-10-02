"""Resource sampling: parsers, nvidia-smi stream, monitor events and graceful nulls."""

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

from audio_transcript.adapters import sysmon
from audio_transcript.application.events import CallbackSink, EventBus
from audio_transcript.application.monitor import ResourceMonitor
from audio_transcript.application.session import RunSession

try:
    from .fakes import make_config
except ImportError:  # discovered with ``-s tests``
    from fakes import make_config

PROC_STAT_1 = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 2 3 4\n"
PROC_STAT_2 = "cpu  150 0 150 900 0 0 0 0 0 0\n"
MEMINFO = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    8192000 kB\n"
STATUS = "Name:\tpython\nVmRSS:\t  204800 kB\n"


class ParserTests(unittest.TestCase):
    def test_nvidia_smi_line(self):
        parsed = sysmon.parse_nvidia_smi_line("0, 87, 4123, 16303, 182.45, 61\n")
        self.assertEqual(
            parsed,
            {
                "gpu_index": 0,
                "gpu_util_percent": 87.0,
                "gpu_mem_used_mb": 4123.0,
                "gpu_mem_total_mb": 16303.0,
                "gpu_power_w": 182.45,
                "gpu_temp_c": 61.0,
            },
        )

    def test_nvidia_smi_unsupported_fields_become_null(self):
        parsed = sysmon.parse_nvidia_smi_line("0, [N/A], 100, 200, [Not Supported], N/A")
        self.assertIsNone(parsed["gpu_util_percent"])
        self.assertIsNone(parsed["gpu_power_w"])
        self.assertIsNone(parsed["gpu_temp_c"])
        self.assertEqual(parsed["gpu_mem_used_mb"], 100.0)

    def test_nvidia_smi_garbage_is_ignored(self):
        for line in ("", "No devices were found", "a,b,c", "x, 1, 2, 3, 4, 5"):
            self.assertIsNone(sysmon.parse_nvidia_smi_line(line), line)

    def test_proc_stat_meminfo_and_vmrss(self):
        self.assertEqual(sysmon.parse_proc_stat(PROC_STAT_1), (800, 1000))
        used, total = sysmon.parse_meminfo(MEMINFO)
        self.assertEqual(total, 16000.0)
        self.assertEqual(used, 8000.0)
        self.assertEqual(sysmon.parse_vmrss(STATUS), 200.0)
        self.assertIsNone(sysmon.parse_proc_stat("nothing"))
        self.assertIsNone(sysmon.parse_meminfo("MemFree: 1 kB"))
        self.assertIsNone(sysmon.parse_vmrss("Name: x"))

    def test_linux_host_computes_cpu_from_deltas(self):
        texts = iter([PROC_STAT_1, MEMINFO, STATUS, PROC_STAT_2, MEMINFO, STATUS])
        host = sysmon.LinuxHost(read=lambda path: next(texts))
        first = host.sample()
        self.assertIsNone(first["cpu_percent"])  # needs two readings
        self.assertEqual(first["ram_total_mb"], 16000.0)
        second = host.sample()
        self.assertEqual(second["cpu_percent"], 50.0)  # 100 busy of 200 elapsed
        self.assertEqual(second["process_rss_mb"], 200.0)

    def test_psutil_is_preferred_when_importable(self):
        class Memory:
            total = 8 * 1024 * 1024 * 1024
            available = 6 * 1024 * 1024 * 1024

        class Info:
            rss = 100 * 1024 * 1024

        class ProcessStub:
            def memory_info(self):
                return Info()

        class FakePsutil:
            Process = ProcessStub

            @staticmethod
            def cpu_percent(interval):
                return 12.34

            @staticmethod
            def virtual_memory():
                return Memory()

        host = sysmon.make_host_sampler(import_module=lambda name: FakePsutil)
        self.assertEqual(host.label, "psutil")
        self.assertEqual(
            host.sample(),
            {
                "cpu_percent": 12.3,
                "ram_used_mb": 2048.0,
                "ram_total_mb": 8192.0,
                "process_rss_mb": 100.0,
            },
        )

    def test_no_host_sampler_on_unknown_platform(self):
        def missing(name):
            raise ImportError(name)

        self.assertIsNone(sysmon.make_host_sampler("plan9", import_module=missing))

    def test_gpu_sampler_absent_without_nvidia_smi(self):
        self.assertIsNone(sysmon.make_gpu_sampler(1.0, which=lambda name: None))
        self.assertIsNotNone(sysmon.make_gpu_sampler(1.0, which=lambda name: "nvidia-smi"))


class FakeProcess:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


class NvidiaStreamTests(unittest.TestCase):
    def test_command_and_latest_reading(self):
        processes = []

        def popen(command, **kwargs):
            processes.append(command)
            return FakeProcess(
                [
                    "0, 10, 1000, 16000, 50.0, 40\n",
                    "1, 99, 2000, 16000, 60.0, 50\n",
                    "0, 87, 4123, 16303, 182.45, 61\n",
                ]
            )

        stream = sysmon.NvidiaSmiStream("nvidia-smi", 0.5, popen=popen, max_restarts=0)
        stream.start()
        deadline = time.monotonic() + 3
        while not stream.sample() and time.monotonic() < deadline:
            time.sleep(0.01)
        reading = stream.sample()
        stream.close()
        self.assertEqual(reading["gpu_util_percent"], 87.0)  # first GPU, latest line
        self.assertEqual(reading["gpu_power_w"], 182.45)
        command = processes[0]
        self.assertIn("--format=csv,noheader,nounits", command)
        self.assertEqual(command[-2:], ["-lms", "500"])
        self.assertTrue(any(part.startswith("--query-gpu=") for part in command))

    def test_restarts_after_the_process_exits(self):
        starts = []

        def popen(command, **kwargs):
            starts.append(1)
            return FakeProcess(["0, 5, 1, 2, 3.0, 4\n"])

        stream = sysmon.NvidiaSmiStream("nvidia-smi", 1.0, popen=popen, max_restarts=1)
        stream.start()
        deadline = time.monotonic() + 5
        while len(starts) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        stream.close()
        self.assertEqual(len(starts), 2)
        self.assertEqual(stream.restarts, 1)

    def test_close_terminates_the_process(self):
        holder = {}
        gate = threading.Event()

        class Blocking(FakeProcess):
            def __init__(self):
                super().__init__([])
                self.stdout = self._lines()

            def _lines(self):
                yield "0, 1, 1, 1, 1, 1\n"
                gate.wait(2)

        def popen(command, **kwargs):
            holder["process"] = Blocking()
            return holder["process"]

        stream = sysmon.NvidiaSmiStream("nvidia-smi", 1.0, popen=popen)
        stream.start()
        deadline = time.monotonic() + 3
        while not stream.sample() and time.monotonic() < deadline:
            time.sleep(0.01)
        stream.close()
        gate.set()
        self.assertTrue(holder["process"].terminated)

    def test_missing_executable_yields_no_reading(self):
        def popen(command, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        stream = sysmon.NvidiaSmiStream("nvidia-smi", 1.0, popen=popen)
        self.assertEqual(stream.sample(), {})
        time.sleep(0.05)
        self.assertEqual(stream.sample(), {})
        stream.close()

    def test_stale_reading_is_dropped(self):
        stream = sysmon.NvidiaSmiStream("nvidia-smi", 1.0, popen=None, stale_after=0.01)
        stream._thread = object()  # no process: only feed() is exercised
        stream.feed("0, 5, 1, 2, 3.0, 4\n")
        self.assertTrue(stream.sample())
        time.sleep(0.03)
        self.assertEqual(stream.sample(), {})


class StubSampler:
    def __init__(self, label, values=None, error=None):
        self.label = label
        self.values = values or {}
        self.error = error
        self.closed = False

    def sample(self):
        if self.error:
            raise self.error
        return dict(self.values)

    def close(self):
        self.closed = True


class MonitorTests(unittest.TestCase):
    def test_sample_merges_samplers_and_nulls_the_rest(self):
        bus = EventBus("r")
        monitor = ResourceMonitor(
            bus,
            [
                StubSampler(
                    "proc", {"cpu_percent": 12.5, "ram_used_mb": 100.0, "ram_total_mb": 200.0}
                ),
                StubSampler("nvidia-smi", {"gpu_util_percent": 80.0, "gpu_mem_used_mb": 1.0}),
            ],
        )
        data = monitor.sample_once()
        self.assertEqual(data["cpu_percent"], 12.5)
        self.assertEqual(data["gpu_util_percent"], 80.0)
        self.assertIsNone(data["gpu_temp_c"])
        self.assertIsNone(data["process_rss_mb"])
        self.assertEqual(data["scope"], "system")
        self.assertEqual(data["source"], {"host": "proc", "gpu": "nvidia-smi"})

    def test_failing_sampler_gives_nulls_and_one_warning(self):
        seen = []
        bus = EventBus("r", [CallbackSink(seen.append)])
        broken = StubSampler("nvidia-smi", error=OSError("gone"))
        monitor = ResourceMonitor(bus, [broken, StubSampler("proc", {"cpu_percent": 1.0})])
        for _ in range(3):
            data = monitor.sample_once()
        self.assertIsNone(data["gpu_util_percent"])
        self.assertEqual(data["cpu_percent"], 1.0)
        warnings = [e for e in seen if e.type == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertNotIn("gone", warnings[0].data["message"])

    def test_no_samplers_still_reports_nulls(self):
        data = ResourceMonitor(EventBus("r"), []).sample_once()
        self.assertTrue(all(data[key] is None for key in ("cpu_percent", "gpu_util_percent")))

    def test_thread_emits_samples_and_stops(self):
        seen = []
        bus = EventBus("r", [CallbackSink(seen.append)])
        sampler = StubSampler("proc", {"cpu_percent": 3.0})
        monitor = ResourceMonitor(bus, [sampler], interval=0.05)
        monitor.start()
        deadline = time.monotonic() + 3
        while sum(e.type == "resource.sample" for e in seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        monitor.stop()
        count = sum(e.type == "resource.sample" for e in seen)
        self.assertGreaterEqual(count, 2)
        time.sleep(0.15)
        self.assertEqual(sum(e.type == "resource.sample" for e in seen), count)
        self.assertTrue(sampler.closed)

    def test_session_runs_the_monitor_and_can_disable_it(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = make_config(root, monitor=enabled, monitor_interval=0.1)
                created = []

                def factory(bus, created=created):
                    created.append(1)
                    return ResourceMonitor(bus, [StubSampler("proc", {"cpu_percent": 1.0})], 0.1)

                session = RunSession(
                    config, command="run", monitor_factory=factory, stderr=io.StringIO()
                )
                session.start()
                time.sleep(0.35)
                session.finish(0, [])
                session.close()
                text = session.paths.events.read_text(encoding="utf-8")
                self.assertEqual(bool(created), enabled)
                self.assertEqual('"type":"resource.sample"' in text, enabled)
                status = session.paths.status.read_text(encoding="utf-8")
                self.assertEqual('"cpu_percent": 1.0' in status, enabled)


if __name__ == "__main__":
    unittest.main()

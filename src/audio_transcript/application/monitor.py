"""Background resource sampling that publishes ``resource.sample`` events.

Samplers are plain objects with ``label``, ``sample() -> dict`` and ``close()``; the OS
specific ones live in ``adapters/sysmon.py``. Sampling problems never fail a run: affected
fields are ``null`` and one ``warning`` event reports the first failure of a sampler.
"""

from __future__ import annotations

import threading
from typing import Protocol

from .events import EventBus
from .progress import RESOURCE_KEYS


class Sampler(Protocol):
    label: str

    def sample(self) -> dict: ...

    def close(self) -> None: ...


class ResourceMonitor:
    def __init__(self, bus: EventBus, samplers: list[Sampler], interval: float = 1.0):
        self.bus = bus
        self.samplers = samplers
        self.interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed: set[int] = set()

    def sample_once(self) -> dict:
        data: dict = {key: None for key in RESOURCE_KEYS}
        sources: dict[str, str] = {}
        for sampler in self.samplers:
            try:
                values = sampler.sample()
            except Exception as exc:
                if id(sampler) not in self._failed:
                    self._failed.add(id(sampler))
                    self.bus.emit(
                        "warning",
                        {
                            "code": "monitor_sampler_failed",
                            "message": f"Resource sampler {sampler.label} failed: "
                            f"{type(exc).__name__}",
                        },
                    )
                continue
            used = False
            for key, value in values.items():
                if key in data and value is not None:
                    data[key] = value
                    used = True
            if used:
                sources[_group(sampler.label)] = sampler.label
        data["source"] = sources
        data["scope"] = "system"
        return data

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            self.sample_once()  # primes CPU deltas; the value is not published
        except Exception:
            pass
        if self._stop.wait(min(self.interval, 0.5)):
            return
        while True:
            try:
                self.bus.emit("resource.sample", self.sample_once())
            except Exception:
                pass
            if self._stop.wait(self.interval):
                break

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        for sampler in self.samplers:
            try:
                sampler.close()
            except Exception:
                pass


def _group(label: str) -> str:
    return "gpu" if "nvidia" in label else "host"

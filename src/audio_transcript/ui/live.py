"""Live ANSI console: ``render`` (pure) and ``LiveConsole`` (an event sink that redraws).

``render(state, width) -> list[str]`` depends only on the ``RunState`` it is given, so it is
tested without a terminal. ``LiveConsole`` redraws in place at most 8 times per second, hides
the cursor while it runs and always restores it.
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable

from ..application.backend_select import header_label
from ..application.events import Event
from ..application.progress import RunState

MAX_FPS = 8
ACTIVITY_ROWS = 8
STAGE_LABELS = (
    ("probe", "probe"),
    ("audio_extraction", "extract"),
    ("speech_detection", "speech"),
    ("chunk_slicing", "slice"),
    ("inference", "transcribe"),
    ("export", "export"),
)

_UNICODE_PROBE = "█░▁▂▃▄▅▆▇✓●○·─°…"

_STYLES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "magenta": "35",
}


class Glyphs:
    def __init__(self, unicode: bool):
        if unicode:
            self.full, self.empty = "█", "░"
            self.spark = "▁▂▃▄▅▆▇█"
            self.done, self.running, self.pending = "✓", "●", "○"
            self.sep, self.rule, self.degree, self.ellipsis = " · ", "─", "°", "…"
        else:
            self.full, self.empty = "#", "-"
            self.spark = ".:-=+*#%"
            self.done, self.running, self.pending = "+", ">", "."
            self.sep, self.rule, self.degree, self.ellipsis = " | ", "-", "", "~"


def supports_unicode(stream=None) -> bool:
    encoding = getattr(stream if stream is not None else sys.stdout, "encoding", None) or "ascii"
    try:
        _UNICODE_PROBE.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def color_enabled(stream=None, environ: dict | None = None) -> bool:
    env = os.environ if environ is None else environ
    if env.get("NO_COLOR"):
        return False
    return True


# ---------------------------------------------------------------------------- formatting


def clock(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    whole = int(max(0, seconds))
    hours, rest = divmod(whole, 3600)
    minutes, sec = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def short_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    if seconds < 10:
        return f"{seconds:.1f}s"
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, sec = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _gb(mb: float | None) -> str:
    return "n/a" if mb is None else f"{mb / 1024:.1f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def sparkline(values, glyphs: Glyphs, length: int, low: float = 0.0, high: float | None = None):
    """Last ``length`` values as block characters scaled between ``low`` and ``high``."""
    series = list(values)[-length:]
    if not series:
        return " " * length
    top = high if high is not None else max(series)
    span = top - low
    chars = []
    for value in series:
        if span <= 0:
            level = 0
        else:
            level = int(round((min(max(value, low), top) - low) / span * (len(glyphs.spark) - 1)))
        chars.append(glyphs.spark[level])
    return "".join(chars).rjust(length)


Segment = tuple[str, str]  # (text, style)


def _paint(text: str, style: str, color: bool) -> str:
    if not color or not style:
        return text
    codes = ";".join(_STYLES[name] for name in style.split("+") if name in _STYLES)
    return f"\x1b[{codes}m{text}\x1b[0m" if codes else text


def _fit(segments: list[Segment], width: int, glyphs: Glyphs, color: bool) -> str:
    """Join segments, truncating on visible width (escape codes are never counted)."""
    total = sum(len(text) for text, _ in segments)
    out: list[str] = []
    if total <= width:
        for text, style in segments:
            out.append(_paint(text, style, color))
        return "".join(out)
    budget = max(0, width - len(glyphs.ellipsis))
    for text, style in segments:
        if budget <= 0:
            break
        piece = text[:budget]
        out.append(_paint(piece, style, color))
        budget -= len(piece)
    out.append(glyphs.ellipsis[: max(0, width)])
    return "".join(out)


# ---------------------------------------------------------------------------- rendering


def render(
    state: RunState,
    width: int,
    *,
    color: bool = True,
    unicode: bool = True,
    activity_rows: int = ACTIVITY_ROWS,
) -> list[str]:
    """One frame as a list of lines, none wider than ``width`` visible characters."""
    glyphs = Glyphs(unicode)
    width = max(30, width) - 1  # the last column stays free so terminals never auto-wrap
    snap = state  # alias: this function only reads the state object
    lines: list[list[Segment]] = []
    job = snap.job
    finished = snap.state == "finished"

    # header
    language = snap.language or "auto"
    left = f" SbobinAI{glyphs.sep}{snap.model or '?'}{glyphs.sep}{language}"
    if snap.parallel_requests:
        left += f"{glyphs.sep}parallel {snap.parallel_requests}"
    runtime_text = header_label(snap.runtime, glyphs.sep)
    runtime_part = f"{glyphs.sep}{runtime_text}" if runtime_text else ""
    right = f"done in {clock(snap.elapsed_s)} " if finished else f"elapsed {clock(snap.elapsed_s)} "
    gap = max(1, width - len(left) - len(runtime_part) - len(right))
    fell_back = bool(snap.runtime and snap.runtime.get("fallback_used"))
    lines.append(
        [
            (left, "bold"),
            (runtime_part, "bold+yellow" if fell_back else "cyan"),
            (" " * gap, ""),
            (right, "dim"),
        ]
    )

    # file
    if job is None:
        lines.append([(" Waiting for the first file...", "dim")])
    else:
        total = job.total or snap.queue_size or "?"
        lines.append(
            [
                (f" File {job.index}/{total}", "bold"),
                (f"  {job.source_name or '?'}", ""),
                (f"  version {job.version}" if job.version else "", "dim"),
            ]
        )

    # stages
    stage_segments: list[Segment] = [(" ", "")]
    for key, label in STAGE_LABELS:
        info = job.stages.get(key) if job is not None else None
        if info is None:
            mark, style, text = glyphs.pending, "dim", label
        elif info["state"] == "running":
            mark, style, text = glyphs.running, "cyan", label
        else:
            seconds = info.get("seconds")
            mark, style = glyphs.done, "green"
            text = (
                f"{label} {short_duration(seconds)}" if seconds is not None else f"{label} cached"
            )
        stage_segments.append((mark, style))
        stage_segments.append((f" {text}  ", "" if info is not None else "dim"))
    lines.append(stage_segments)

    # progress bar
    progress = job.progress if job is not None else None
    done = progress["done_chunks"] if progress else 0
    total_chunks = progress["total_chunks"] if progress else 0
    percent = progress["percent"] if progress else 0.0
    tail = f" {done}/{total_chunks} {percent:3.0f}%"
    if progress and progress.get("eta_s") is not None:
        tail += f"  ETA {short_duration(progress['eta_s'])}"
    elif progress and done < total_chunks:
        tail += "  ETA ..."
    if progress and progress.get("throughput_x"):
        tail += f"  x{progress['throughput_x']:g} realtime"
    prefix = " Chunks ["
    bar_width = max(8, width - len(prefix) - len(tail) - 1)
    filled = int(round(bar_width * min(100.0, percent) / 100.0))
    lines.append(
        [
            (prefix, "bold"),
            (glyphs.full * filled, "green"),
            (glyphs.empty * (bar_width - filled), "dim"),
            ("]", "bold"),
            (tail, ""),
        ]
    )

    # counters
    counters = progress or {}
    failed = counters.get("failed", 0)
    flight = (job.in_flight if job is not None else []) or []
    flight_text = " ".join(f"#{index + 1}" for index in flight) if flight else "-"
    sep = glyphs.sep
    lines.append(
        [
            (f" ok {counters.get('ok', 0)}", "green"),
            (f"{sep}no speech {counters.get('no_speech', 0)}", ""),
            (f"{sep}failed {failed}", "red+bold" if failed else ""),
            (
                f"{sep}fallbacks {counters.get('fallbacks', 0)}",
                "yellow" if counters.get("fallbacks") else "",
            ),
            (f"{sep}splits {counters.get('splits', 0)}", ""),
            (f"{sep}in flight {flight_text}", "cyan" if flight else "dim"),
        ]
    )

    # audio / tokens / confidence
    prepared = job.prepared if job is not None else None
    audio_done = counters.get("audio_done_s")
    audio_total = counters.get("audio_total_s")
    speech = prepared.get("speech_seconds") if prepared else None
    confidence = counters.get("confidence_mean")
    tokens = counters.get("tokens")
    audio_text = f" audio {short_duration(audio_done) if audio_done is not None else '?'}"
    audio_text += f" / {short_duration(audio_total) if audio_total is not None else '?'}"
    if speech is not None:
        audio_text += f" (speech {short_duration(speech)})"
    audio_text += f"{sep}tokens {int(tokens):,}" if tokens is not None else f"{sep}tokens n/a"
    audio_text += (
        f"{sep}confidence {confidence:.2f}" if confidence is not None else f"{sep}confidence n/a"
    )
    lines.append([(audio_text, "")])

    # resources
    res = snap.resources
    hist = snap.history
    spark = 10 if width >= 99 else 6

    def row(*parts: tuple[str, str, str, str]) -> list[Segment]:
        segments: list[Segment] = [(" ", "")]
        for label, spark_text, value_text, style in parts:
            segments.append((f"{label} ", "bold"))
            segments.append((spark_text, style))
            segments.append((f" {value_text}  ", ""))
        return segments

    ram_total = res.get("ram_total_mb") if res else None
    vram_total = res.get("gpu_mem_total_mb") if res else None
    cpu = res.get("cpu_percent") if res else None
    gpu = res.get("gpu_util_percent") if res else None
    ram = res.get("ram_used_mb") if res else None
    vram = res.get("gpu_mem_used_mb") if res else None
    power = res.get("gpu_power_w") if res else None
    temp = res.get("gpu_temp_c") if res else None
    lines.append(
        row(
            ("CPU", sparkline(hist["cpu_percent"], glyphs, spark, 0, 100), _pct(cpu), "cyan"),
            (
                "RAM",
                sparkline(hist["ram_used_mb"], glyphs, spark, 0, ram_total),
                f"{_gb(ram)}/{_gb(ram_total)} GB",
                "cyan",
            ),
        )
    )
    power_text = "n/a" if power is None else f"{power:.0f} W"
    temp_text = "n/a" if temp is None else f"{temp:.0f} {glyphs.degree}C"
    hot = temp is not None and temp >= 83
    lines.append(
        row(
            ("GPU", sparkline(hist["gpu_util_percent"], glyphs, spark, 0, 100), _pct(gpu), "green"),
            (
                "VRAM",
                sparkline(hist["gpu_mem_used_mb"], glyphs, spark, 0, vram_total),
                f"{_gb(vram)}/{_gb(vram_total)} GB",
                "green",
            ),
            ("PWR", sparkline(hist["gpu_power_w"], glyphs, spark, 0), power_text, "yellow"),
            (
                "TEMP",
                sparkline(hist["gpu_temp_c"], glyphs, spark, 0, 100),
                temp_text,
                "red" if hot else "magenta",
            ),
        )
    )

    # activity
    scope = (res or {}).get("scope")
    title = f" activity{' (resources are system-wide)' if scope == 'system' else ''} "
    rule = glyphs.rule * max(0, width - len(title) - 3)
    lines.append([(f"{glyphs.rule * 2}{title}{rule}", "dim")])
    recent = list(snap.activity)[-activity_rows:]
    for entry in recent:
        level = entry["level"]
        style = "red" if level == "error" else "yellow" if level == "warning" else ""
        lines.append([(f" {clock(entry['elapsed_s'])}  ", "dim"), (entry["text"], style)])
    for _ in range(activity_rows - len(recent)):
        lines.append([("", "")])
    return [_fit(segments, width, glyphs, color) for segments in lines]


# ---------------------------------------------------------------------------- live sink


class LiveConsole:
    """Event sink that keeps a ``RunState`` and redraws it in place (at most ``MAX_FPS``)."""

    def __init__(
        self,
        stream=None,
        *,
        color: bool | None = None,
        unicode: bool | None = None,
        width: Callable[[], int] | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        max_fps: int = MAX_FPS,
    ):
        self.stream = stream if stream is not None else sys.stdout
        self.color = color_enabled(self.stream) if color is None else color
        self.unicode = supports_unicode(self.stream) if unicode is None else unicode
        self._width = width or (lambda: shutil.get_terminal_size((100, 30)).columns)
        self._clock = clock_fn
        self._started = clock_fn()
        self._interval = 1.0 / max_fps
        self.state = RunState()
        self._lock = threading.RLock()
        self._dirty = True
        self._height = 0
        self._cursor_hidden = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._atexit = self._restore_cursor

    # -- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._write("\x1b[?25l")
        self._cursor_hidden = True
        atexit.register(self._atexit)
        self._thread = threading.Thread(target=self._loop, name="live-console", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.state.tick(self._clock() - self._started)
            self.draw()

    def handle(self, event: Event) -> None:
        if self._thread is None:
            self.start()
        self.state.apply(event)
        with self._lock:
            self._dirty = True
        if event.type in {"run.finished"}:
            self.draw()

    def draw(self) -> None:
        with self._lock:
            if self._closed or not self._dirty:
                return
            self._dirty = False
            self.state.tick(self._clock() - self._started)
            try:
                frame = render(
                    self.state,
                    self._width(),
                    color=self.color,
                    unicode=self.unicode,
                )
            except Exception:  # a rendering bug must never break the run
                return
            out = []
            if self._height:
                out.append(f"\x1b[{self._height}F")
            for line in frame:
                out.append(f"{line}\x1b[K\n")
            out.append("\x1b[J")
            self._height = len(frame)
            self._write("".join(out))

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except UnicodeEncodeError:
            self.unicode = False
        except (OSError, ValueError):
            pass

    def _restore_cursor(self) -> None:
        if self._cursor_hidden:
            self._write("\x1b[?25h")
            self._cursor_hidden = False

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            self._dirty = True
            self.draw()  # the final frame stays on screen
            self._closed = True
            self._restore_cursor()
        try:
            atexit.unregister(self._atexit)
        except Exception:
            pass

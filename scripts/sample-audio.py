"""Select non-silent random central media excerpts for local inference checks."""

import argparse
import json
import math
import random
import re
import shutil
import subprocess
from pathlib import Path

from audio_transcript.adapters.ffmpeg import FFmpegProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(".local/samples/input"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--min-peak-db", type=float, default=-45)
    parser.add_argument("--min-mean-db", type=float, default=-60)
    args = parser.parse_args()
    if (
        args.count < 1
        or args.attempts < args.count
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
    ):
        parser.error("count/seconds must be positive and attempts must be at least count")
    if shutil.which("ffmpeg") is None:
        parser.error("FFmpeg is required on PATH")
    source = args.source.resolve()
    info = FFmpegProcessor().probe(source)
    seconds = min(args.seconds, info.duration)
    lower = min(info.duration * 0.35, info.duration - seconds)
    upper = max(lower, min(info.duration * 0.65, info.duration - seconds))
    generator = random.Random(args.seed)
    args.destination.mkdir(parents=True, exist_ok=True)
    accepted: list[dict] = []
    checked: list[dict] = []
    for attempt in range(args.attempts):
        start = generator.uniform(lower, upper)
        if any(abs(start - previous["start_seconds"]) < seconds for previous in accepted):
            continue
        measurement = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(source),
                "-t",
                str(seconds),
                "-map",
                "0:a:0",
                "-vn",
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if measurement.returncode:
            raise RuntimeError(
                "Could not measure the media audio signal: " + measurement.stderr[-1200:]
            )
        levels = {}
        for key in ("mean_volume", "max_volume"):
            match = re.search(rf"{key}:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB", measurement.stderr)
            levels[key] = float(match.group(1)) if match else float("-inf")
        has_signal = (
            levels["max_volume"] >= args.min_peak_db and levels["mean_volume"] >= args.min_mean_db
        )
        evidence = {
            "attempt": attempt + 1,
            "start_seconds": start,
            "end_seconds": start + seconds,
            "mean_dbfs": levels["mean_volume"] if math.isfinite(levels["mean_volume"]) else None,
            "peak_dbfs": levels["max_volume"] if math.isfinite(levels["max_volume"]) else None,
            "has_signal": has_signal,
        }
        checked.append(evidence)
        print(
            f"Window {start:.1f}–{start + seconds:.1f}s: mean {levels['mean_volume']:.1f} dBFS, "
            f"peak {levels['max_volume']:.1f} dBFS; {'signal detected' if has_signal else 'rejected as quiet'}"
        )
        if not has_signal:
            continue
        target = args.destination / f"central_sample_{len(accepted) + 1:02d}.wav"
        extraction = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(source),
                "-t",
                str(seconds),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if extraction.returncode:
            raise RuntimeError("Could not extract sample: " + extraction.stderr[-1200:])
        evidence["sample_file"] = str(target.resolve())
        accepted.append(evidence)
        if len(accepted) == args.count:
            break
    report = {
        "source": str(source),
        "source_duration_seconds": info.duration,
        "sample_seconds": seconds,
        "selection_region": [lower, upper],
        "random_seed": args.seed,
        "thresholds_dbfs": {"peak": args.min_peak_db, "mean": args.min_mean_db},
        "accepted": accepted,
        "checked": checked,
        "note": "Audio energy indicates signal; it does not establish speech content or transcription accuracy.",
    }
    (args.destination / "sampling-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if len(accepted) != args.count:
        print(f"Only {len(accepted)} of {args.count} non-silent samples were found.")
        return 1
    print(f"Prepared {len(accepted)} central samples with audio signal in {args.destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

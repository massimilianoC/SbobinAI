"""Run a real FFmpeg + synthetic backend smoke test in ignored local storage."""

import json
import subprocess
import sys
import wave
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    root = project / ".local" / "smoke"
    inputs = root / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    fixture = inputs / "synthetic sample.wav"
    if not fixture.exists():
        with wave.open(str(fixture), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 40000)
    command = [
        sys.executable,
        "-m",
        "audio_transcript",
        "run",
        "--backend",
        "mock",
        "--input-dir",
        str(inputs),
        "--process-dir",
        str(root / "process"),
        "--output-dir",
        str(root / "output"),
        "--chunk-seconds",
        "1",
        "--force",
    ]
    result = subprocess.run(command, cwd=project, check=False)
    if result.returncode:
        return result.returncode
    expected = {
        "transcript.json",
        "transcript.txt",
        "transcript.md",
        "transcript.srt",
        "transcript.vtt",
        "report.json",
    }
    for output in (root / "output").iterdir():
        if not output.is_dir():
            continue
        if not expected.issubset({path.name for path in output.iterdir()}):
            raise RuntimeError("Smoke output is incomplete.")
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "mock_completed":
            raise RuntimeError("Mock output was not identified as synthetic.")
    print("Smoke test passed: actual FFmpeg preparation and explicitly synthetic transcripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

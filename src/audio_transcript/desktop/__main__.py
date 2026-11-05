"""Launch with python -m audio_transcript.desktop --config config.local.toml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="SbobinAI guided desktop transcription")
    parser.add_argument("--config", type=Path, default=Path("config.local.toml"))
    args = parser.parse_args()
    try:
        import webview
    except ImportError:
        print(
            "Desktop dependencies are missing. Install them in the project environment with: "
            f'"{sys.executable}" -m pip install -e ".[desktop]"',
            file=sys.stderr,
        )
        return 1
    from .bridge import DesktopBridge

    try:
        bridge = DesktopBridge(args.config)
    except (OSError, ValueError) as exc:
        print(f"Could not open the desktop app: {exc}", file=sys.stderr)
        if not args.config.exists():
            print(
                "Create the configuration by following the README installation "
                "(setup), or pass an existing file with --config PATH.",
                file=sys.stderr,
            )
        return 1
    window = webview.create_window(
        "SbobinAI",
        str(Path(__file__).with_name("assets") / "index.html"),
        js_api=bridge,
        width=1100,
        height=830,
        min_size=(820, 700),
        background_color="#f5f5f0",
    )
    bridge._window = window
    window.events.closing += bridge.can_close
    window.events.closed += bridge.shutdown
    try:
        webview.start()
    finally:
        bridge.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

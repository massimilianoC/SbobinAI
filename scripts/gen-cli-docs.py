"""Regenerate the committed CLI documentation from the single source (clidoc).

    python scripts/gen-cli-docs.py           # write the files
    python scripts/gen-cli-docs.py --check   # exit 1 when a committed file is stale

Equivalent for the reference alone: ``sbobinai man --format markdown > docs/cli-reference.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audio_transcript import climan  # noqa: E402

TARGETS = {
    ROOT / "docs" / "cli-reference.md": lambda: climan.markdown(),
    ROOT / "skills" / "sbobinai" / "SKILL.md": lambda: climan.skill_text(),
    ROOT / "src" / "audio_transcript" / "skills" / "SKILL.md": lambda: climan.skill_text(),
}


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale = []
    for path, render in TARGETS.items():
        text = render()
        current = path.read_bytes().decode("utf-8") if path.exists() else None
        if current == text:
            continue
        stale.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote {path.relative_to(ROOT)}")
    if check and stale:
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}; run python scripts/gen-cli-docs.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

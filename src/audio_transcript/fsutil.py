"""Small filesystem helpers shared by application code and adapters."""

from __future__ import annotations

import os
import time

# Windows refuses to replace a file while another process (antivirus, indexer,
# file watcher, a reader of status files) holds it open without delete sharing.
# Such locks are brief, so a short retry turns a spurious job failure into a
# few milliseconds of delay. Other errors are raised immediately.
REPLACE_ATTEMPTS = 10
REPLACE_FIRST_DELAY = 0.02
REPLACE_MAX_DELAY = 0.4


def replace_with_retry(source: os.PathLike | str, destination: os.PathLike | str) -> None:
    """``os.replace`` that retries transient ``PermissionError`` (about 2 s at most)."""
    delay = REPLACE_FIRST_DELAY
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, REPLACE_MAX_DELAY)

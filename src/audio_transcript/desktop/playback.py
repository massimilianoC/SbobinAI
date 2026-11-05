"""Loopback media transport for review playback.

Serves exactly one allow-listed file at a time on 127.0.0.1 under a random token,
with HTTP Range support so the player can seek in large files. The request path
never selects a file, and nothing is written to stdout.
"""

from __future__ import annotations

import mimetypes
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK = 256 * 1024


def parse_range(header: str | None, size: int) -> tuple[int, int] | None | str:
    """Return (first, last), None for no usable single range, or "bad" if unsatisfiable."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    first, _, last = header[6:].strip().partition("-")
    try:
        if first == "":
            length = int(last)
            if length <= 0:
                return "bad"
            return max(0, size - length), size - 1
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    except ValueError:
        return None
    return (start, end) if 0 <= start <= end and start < size else "bad"


class PlaybackServer:
    def __init__(self) -> None:
        self._token = secrets.token_urlsafe(24)
        self._lock = threading.Condition()
        self._path: Path | None = None
        self._generation = 0
        self._active = 0
        self._connections: set = set()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_HEAD(self) -> None:
                self._send(False)

            def do_GET(self) -> None:
                self._send(True)

            def _send(self, body: bool) -> None:
                served = (
                    outer._acquire(self.connection)
                    if self.path.partition("?")[0] == outer._route
                    else None
                )
                if served is None:
                    return self.send_error(404)
                path, generation = served
                try:
                    with path.open("rb") as stream:
                        size = path.stat().st_size
                        requested = parse_range(self.headers.get("Range"), size)
                        if requested == "bad":
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{size}")
                            self.send_header("Content-Length", "0")
                            return self.end_headers()
                        first, last = requested or (0, size - 1)
                        self.send_response(206 if requested else 200)
                        if requested:
                            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
                        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                        self.send_header("Content-Type", kind)
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(max(0, last - first + 1)))
                        self.end_headers()
                        if not body or size == 0:
                            return
                        stream.seek(first)
                        remaining = last - first + 1
                        while remaining > 0 and outer._generation == generation:
                            data = stream.read(min(CHUNK, remaining))
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                except OSError:
                    pass  # file removed or the player closed the connection
                finally:
                    outer._finish(self.connection)

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, *_args) -> None:
                pass

        self._route = f"/{self._token}/media"
        self._httpd = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}{self._route}"

    def _acquire(self, connection) -> tuple[Path, int] | None:
        with self._lock:
            if self._path is None:
                return None
            self._active += 1
            self._connections.add(connection)
            return self._path, self._generation

    def _finish(self, connection) -> None:
        with self._lock:
            self._active -= 1
            self._connections.discard(connection)
            self._lock.notify_all()

    def serve(self, path: Path) -> None:
        self.release()
        with self._lock:
            self._path = path

    def release(self) -> None:
        """Stop serving and wait until no request still holds the file open."""
        with self._lock:
            self._path = None
            self._generation += 1
            for connection in list(self._connections):
                try:  # unblock a write that waits for a player that stopped reading
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self._lock.wait_for(lambda: self._active == 0, timeout=5)

    def close(self) -> None:
        self.release()
        self._httpd.shutdown()
        self._httpd.server_close()

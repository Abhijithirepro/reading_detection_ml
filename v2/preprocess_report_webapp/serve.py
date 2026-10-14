#!/usr/bin/env python3
"""
serve.py — serves this folder (the pre-scored reading-detector page) for servers
without nginx / Apache. Python 3.8+ standard library only: no pip install.

Unlike `python3 -m http.server` it answers HTTP Range requests, so the videos can
be seeked (clicking a reading segment jumps there), and it never lists folders.

    python3 serve.py                          # http://127.0.0.1:8002  (this machine only)
    python3 serve.py --host 0.0.0.0           # reachable from other machines — there is NO login
    python3 serve.py --host 0.0.0.0 --port 80
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
CHUNK = 256 * 1024


class Handler(SimpleHTTPRequestHandler):
    quiet = False
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".webm": "video/webm", ".mp4": "video/mp4", ".json": "application/json",
                      ".js": "text/javascript", ".css": "text/css", ".csv": "text/csv"}

    def list_directory(self, path):
        self.send_error(HTTPStatus.FORBIDDEN, "Folder listing is disabled")
        return None

    def send_head(self):
        self._left = None
        path = self.translate_path(self.path)
        rng = RANGE.match((self.headers.get("Range") or "").strip())
        if not rng or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = rng.groups()
        if first == "":                                   # "bytes=-500": the last 500 bytes
            start, end = max(0, size - int(last or 0)), size - 1
        else:
            start, end = int(first), min(int(last), size - 1) if last else size - 1
        if start >= size or start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._left = end - start + 1
        return f

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self._left is None:
                return super().copyfile(source, outputfile)
            left = self._left
            while left > 0:
                chunk = source.read(min(CHUNK, left))
                if not chunk:
                    break
                outputfile.write(chunk)
                left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass                                          # the browser stopped this video request

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the pre-scored reading-detector page.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 0.0.0.0 exposes it on the network (there is no login)")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--quiet", action="store_true", help="don't log every request")
    args = ap.parse_args(argv)
    if not (ROOT / "data" / "index.json").exists():
        print(f"warning: {ROOT / 'data'} is empty — the page will show no candidates", file=sys.stderr)
    Handler.quiet = args.quiet
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(ROOT)))
    print(f"Reading detector (pre-scored) — http://{args.host}:{args.port}   serving {ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

import json
import socket
import struct
import threading
import time
from typing import Dict, Optional

class IWR6843Remote:
    """
    Connects to the Pi bridge (length-prefixed JSON over TCP) in a background thread.
    Caches the latest full frame (trackData, pointCloud optional). Auto-reconnects.
    """
    def __init__(self, host: str = "raspberrypi.local", port: int = 55000,
                 conn_timeout: float = 5.0,
                 read_timeout: float = 3.0,
                 idle_reset_s: float = 5.0):
        self.host = host
        self.port = port
        self.conn_timeout = conn_timeout
        self.read_timeout = read_timeout
        self.idle_reset_s = idle_reset_s

        self._sock: Optional[socket.socket] = None
        self._stop = False
        self._latest: Dict = {}
        self._lock = threading.Lock()

        self._thr = threading.Thread(target=self._reader_loop, daemon=True)
        self._thr.start()

    # Public API — mirrors your local interface
    def get_targets(self) -> Dict:
        """Return the latest received frame (or an empty one if nothing yet)."""
        with self._lock:
            if self._latest:
                return dict(self._latest)
            return {"trackData": [], "frameNum": 0}

    def get_latest_frame(self) -> Dict:
        """Alias if other parts of the pipeline prefer this name."""
        return self.get_targets()

    def close(self):
        self._stop = True
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

    # Internals
    def _reader_loop(self):
        backoff = 0.25
        while not self._stop:
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.conn_timeout)
                # Use per-read timeout to detect silent stalls
                s.settimeout(self.read_timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Optional TCP keepalive (platform dependent)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception:
                    pass
                self._sock = s
                backoff = 0.25  # reset backoff after successful connect

                last_ok = time.time()
                while not self._stop:
                    hdr = self._recv_exact(s, 4)
                    if not hdr:
                        raise RuntimeError("connection lost (no header)")
                    if hdr[:1] in (b'{', b'[', b' ', b'\t'):
                        # Newline-delimited JSON — read the rest of the line
                        line_rest = self._recv_until(s, b'\n')
                        if line_rest is None:
                            raise RuntimeError("connection lost (no line)")
                        payload = (hdr + line_rest).rstrip(b'\r\n')
                    else:
                        # Length-prefixed JSON
                        (length,) = struct.unpack("!I", hdr)
                        if length <= 0 or length > 64 * 1024 * 1024:
                            # Not a sensible length; assume this is actually a JSON line stream
                            line_rest = self._recv_until(s, b'\n')
                            if line_rest is None:
                                raise RuntimeError("connection lost (no line)")
                            payload = (hdr + line_rest).rstrip(b'\r\n')
                        else:
                            payload = self._recv_exact(s, length)
                            if not payload:
                                raise RuntimeError("connection lost (no payload)")

                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue  # skip malformed

                    # Accept only proper frames; heartbeats are ignored
                    if isinstance(obj, dict) and (
                        "trackData" in obj or "frameNum" in obj or obj.get("_schema") == "iwr6843.full.v1"
                    ):
                        with self._lock:
                            self._latest = obj
                        last_ok = time.time()
                    else:
                        # heartbeat or unknown — update idle timer anyway
                        last_ok = time.time()

                    # Proactive idle reset: if no valid data for a while, reconnect
                    if (time.time() - last_ok) > self.idle_reset_s:
                        raise TimeoutError("idle timeout waiting for frames")

            except Exception:
                # Connection failed or dropped — exponential backoff up to 3s
                try:
                    if self._sock:
                        self._sock.close()
                except Exception:
                    pass
                self._sock = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 3.0)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    @staticmethod
    def _recv_until(sock: socket.socket, delim: bytes, max_bytes: int = 8 * 1024 * 1024) -> Optional[bytes]:
        buf = b""
        while True:
            ch = sock.recv(1)
            if not ch:
                return None
            buf += ch
            if buf.endswith(delim):
                return buf
            if len(buf) > max_bytes:
                return None

# Alias so your pipeline can import this as the "interface" unchanged:
IWR6843Interface = IWR6843Remote


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="raspberrypi.local")
    ap.add_argument("--port", type=int, default=55000)
    args = ap.parse_args()

    radar = IWR6843Remote(args.host, args.port)
    print("[PC] Connected reader. Ctrl+C to exit.")
    last = -1
    try:
        while True:
            fr = radar.get_targets()
            fn = fr.get("frameNum", -1)
            tracks = len(fr.get("trackData", []))
            # only print when frame number advances to avoid flooding
            if fn != last:
                print(f"[PC] frame={fn} tracks={tracks}")
                last = fn
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        radar.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, socket, signal, logging, threading, argparse, subprocess
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import psutil  # optional
except Exception:
    psutil = None

from iwr6843_interface import IWR6843Interface  # your existing interface

# ──────────────────────────────────────────────────────────────────────────────
# Defaults from env (overridable by CLI)
# ──────────────────────────────────────────────────────────────────────────────
DEF_HOST           = os.environ.get("PI_BRIDGE_HOST", "0.0.0.0")
DEF_PORT           = int(os.environ.get("PI_BRIDGE_PORT", "55000"))
DEF_HEALTH_PORT    = int(os.environ.get("PI_BRIDGE_HEALTH_PORT", "6060"))
DEF_SEND_HEATMAPS  = bool(int(os.environ.get("PI_BRIDGE_SEND_HEATMAPS", "0")))
DEF_TARGETS_ONLY   = bool(int(os.environ.get("PI_BRIDGE_TARGETS_ONLY", "1")))
DEF_MAX_PC_POINTS  = int(os.environ.get("PI_BRIDGE_MAX_PC_POINTS", "0"))
DEF_SINGLE_CLIENT  = True  # always single-client for this bridge

# ──────────────────────────────────────────────────────────────────────────────
# Logging & globals
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PI] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("pi_radar_bridge")
_shutdown = threading.Event()

_uptime_start = time.time()
_cpu_percent: Optional[float] = None
_temp_c: Optional[float] = None
_health_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────────────
def _json_sanitise(x):
    if isinstance(x, (np.generic,)): return x.item()
    if isinstance(x, np.ndarray):    return x.tolist()
    if isinstance(x, (bytes, bytearray)): return x.decode("utf-8", "ignore")
    if isinstance(x, dict):  return {str(k): _json_sanitise(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [_json_sanitise(v) for v in x]
    return x

def _pick(frame: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k not in frame: continue
        v = frame[k]
        if v is None: continue
        try:
            if isinstance(v, np.ndarray):
                if v.size > 0: return v
            elif isinstance(v, (list, tuple, dict, set)):
                if len(v) > 0: return v
            else:
                return v
        except Exception:
            continue
    return default

# ──────────────────────────────────────────────────────────────────────────────
# Compact frame (same spirit as your snippet)
# ──────────────────────────────────────────────────────────────────────────────
def _compact_frame(frame: Dict[str, Any],
                   targets_only: bool,
                   include_heatmaps: bool,
                   max_pc_points: int) -> Dict[str, Any]:
    # tolerant read of upstream track list
    td = frame.get("trackData")
    if not td:
        for alt in ("pre_track", "preTrack", "tracks", "targets", "objects"):
            cand = frame.get(alt)
            if cand:
                td = cand
                break
    if isinstance(td, np.ndarray):
        td = td.tolist()
    if td is None:
        td = []

    out: Dict[str, Any] = {"trackData": td}

    if not targets_only:
        pc = _pick(frame, "pointCloud", "point_cloud", "pointcloud")
        if isinstance(pc, np.ndarray): pc = pc.tolist()
        if isinstance(pc, list) and max_pc_points and len(pc) > max_pc_points:
            pc = pc[:max_pc_points]
        if pc: out["pointCloud"] = pc

    # small health/footer when present
    for k in ("frameNum","numDetectedObj","numTLVs","dopplerResolutionMps",
              "rangeResolutionMeters","frameTimeTriple","v_unamb_ms"):
        if k in frame:
            out[k] = _json_sanitise(frame[k])

    if include_heatmaps:
        out["has_range_azimuth_heatmap"] = bool(_pick(
            frame,"range_azimuth_heatmap","RANGE_AZIMUTH_HEATMAP","rangeAzimuthHeatMap","azimuthHeatMap"
        ))
        out["has_range_doppler_heatmap"] = bool(_pick(
            frame,"range_doppler_heatmap","RANGE_DOPPLER_HEAT_MAP","rangeDopplerHeatMap","dopplerHeatMap"
        ))
        out["has_azimuth_elevation_heatmap"] = bool(_pick(
            frame,"azimuth_elevation_heatmap","AZIMUTH_ELEVATION_HEATMAP","azimuthElevationHeatMap"
        ))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Radar reader thread (your logic, with tiny robustness tweaks)
# ──────────────────────────────────────────────────────────────────────────────
class RadarThread(threading.Thread):
    def __init__(self, targets_only: bool, send_heatmaps: bool, max_pc_points: int):
        super().__init__(name="RadarReader", daemon=True)
        self.targets_only = targets_only
        self.send_heatmaps = send_heatmaps
        self.max_pc_points = max_pc_points

        self.radar: Optional[IWR6843Interface] = None
        self.latest: Dict[str, Any] = {}
        self.last_ok = 0.0
        self.frame_idx = 0  # bridge-side frame counter

    def run(self):
        try:
            self.radar = IWR6843Interface()
            log.info("Connected to IWR6843 interface")
        except Exception as e:
            log.error(f"Failed to init radar interface: {e}")
            return

        while not _shutdown.is_set():
            try:
                frame = self.radar.get_targets() or {}
                if isinstance(frame, dict) and frame:
                    out = _compact_frame(frame, self.targets_only, self.send_heatmaps, self.max_pc_points)
                    # aliases for clients
                    td = out.get("trackData", []) or []
                    if isinstance(td, np.ndarray): td = td.tolist()
                    n = len(td)
                    out["targets"] = td
                    out["tracks"] = td
                    out["numTargets"] = n
                    out["numDetectedTracks"] = n

                    self.frame_idx += 1
                    out["frame"] = self.frame_idx
                    out["t"] = time.time()

                    # expose unambiguous velocity if available
                    try:
                        vu = getattr(self.radar, "v_unamb_ms", None)
                        if vu is not None:
                            out["v_unamb_ms"] = float(vu)
                    except Exception:
                        pass

                    self.latest = out
                    self.last_ok = time.time()
                else:
                    time.sleep(0.01)
            except Exception as e:
                log.info(f"[RADAR] read error: {e}")
                time.sleep(0.05)

            if time.time() - self.last_ok > 3.0:
                log.info("No live stream detected → (re)applying cfg.")
                time.sleep(0.5)

# ──────────────────────────────────────────────────────────────────────────────
# Single-client TCP server (with rate-limit via --hz)
# ──────────────────────────────────────────────────────────────────────────────
class SingleClientServer(threading.Thread):
    def __init__(self, radar: RadarThread, host: str, port: int, hz: float):
        super().__init__(name="TCPServer", daemon=True)
        self.radar = radar
        self.host = host
        self.port = port
        self.min_period = 1.0 / max(1e-3, hz)
        self.sock: Optional[socket.socket] = None
        self.client: Optional[socket.socket] = None
        self.client_addr = None
        self._last_sent_ts = 0.0
        self._last_sent_wall = 0.0

    def _close_client(self):
        if self.client:
            try: self.client.shutdown(socket.SHUT_RDWR)
            except Exception: pass
            try: self.client.close()
            except Exception: pass
        self.client = None
        self.client_addr = None

    def _send_line(self, data: Dict[str, Any]):
        if not self.client: return
        try:
            line = json.dumps(_json_sanitise(data), separators=(",", ":")).encode("utf-8") + b"\n"
            self.client.sendall(line)
        except (BrokenPipeError, ConnectionResetError):
            log.info("Client disconnected")
            self._close_client()
        except Exception as e:
            log.info(f"Send error: {e}")
            self._close_client()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            self.sock = s
            log.info(f"Listening on {self.host}:{self.port} (single client)")
            s.settimeout(0.5)

            while not _shutdown.is_set():
                if not self.client:
                    try:
                        conn, addr = s.accept()
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        try:
                            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                        except Exception:
                            pass
                        conn.settimeout(0.25)  # prevent stalls on slow readers
                        self.client = conn
                        self.client_addr = addr
                        self._last_sent_ts = 0.0
                        self._last_sent_wall = 0.0
                        log.info(f"Client connected: {addr}")
                    except socket.timeout:
                        pass
                    except Exception as e:
                        log.info(f"Accept error: {e}")
                        time.sleep(0.1)

                if self.client and self.radar.latest:
                    now = time.time()
                    t = self.radar.latest.get("t", 0.0)
                    if t and (t != self._last_sent_ts) and (now - self._last_sent_wall >= self.min_period):
                        self._send_line(self.radar.latest)
                        self._last_sent_ts = t
                        self._last_sent_wall = now
                time.sleep(0.005)
            self._close_client()

    def client_count(self) -> int:
        return 1 if self.client is not None else 0

# ──────────────────────────────────────────────────────────────────────────────
# Health helpers (CPU/Temp samplers) — background, non-blocking
# ──────────────────────────────────────────────────────────────────────────────
def _cpu_via_procstat(sample_ms: int = 200) -> Optional[float]:
    try:
        def read() -> Optional[Tuple[int,int]]:
            with open("/proc/stat", "r") as f:
                line = f.readline()
            p = line.split()
            if not p or p[0] != "cpu": return None
            vals = list(map(int, p[1:]))
            idle = vals[3] + vals[4]  # idle + iowait
            total = sum(vals)
            return total, idle

        a = read()
        if not a: return None
        time.sleep(sample_ms / 1000.0)
        b = read()
        if not b: return None
        totald = b[0] - a[0]
        idled = b[1] - a[1]
        if totald <= 0: return None
        return max(0.0, min(100.0, (1.0 - (idled/float(totald))) * 100.0))
    except Exception:
        return None

def _read_pi_temperature_c() -> Optional[float]:
    try:
        p = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(p):
            with open(p, "r") as f:
                return int(f.read().strip()) / 1000.0
    except Exception:
        pass
    try:
        out = subprocess.check_output(["/usr/bin/vcgencmd", "measure_temp"],
                                      stderr=subprocess.DEVNULL, text=True)
        if "temp=" in out:
            v = out.split("temp=")[1].split("'")[0]
            return float(v)
    except Exception:
        pass
    try:
        if psutil:
            temps = psutil.sensors_temperatures()
            if temps:
                for arr in temps.values():
                    if arr:
                        return float(arr[0].current)
    except Exception:
        pass
    return None

def _resource_sampler_loop(period_s: float = 2.0):
    global _cpu_percent, _temp_c
    if psutil:
        try: psutil.cpu_percent(interval=0.3)
        except Exception: pass
    while not _shutdown.is_set():
        # CPU
        val = None
        if psutil:
            try:
                val = float(psutil.cpu_percent(interval=None))
                if not (0.0 <= val <= 100.0): val = None
            except Exception:
                val = None
        if val is None:
            val = _cpu_via_procstat(200)
        # Temp
        t = _read_pi_temperature_c()
        with _health_lock:
            _cpu_percent = None if val is None else round(val, 1)
            _temp_c = None if t is None else round(float(t), 1)
        time.sleep(period_s)

def start_resource_sampler():
    threading.Thread(target=_resource_sampler_loop, name="resource-sampler", daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# Health HTTP (emits keys the app expects: temperature, cpu_load)
# ──────────────────────────────────────────────────────────────────────────────
def start_health_http(bind_host: str, port: int, radar: RadarThread, server: SingleClientServer):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_response(404); self.end_headers(); return
            now = time.time()
            age = now - (radar.last_ok or 0.0)
            radar_ok = age < 3.0
            with _health_lock:
                cpu = _cpu_percent
                temp = _temp_c

            payload = {
                "ts": now,
                "uptime_s": now - _uptime_start,
                "heartbeat": int(radar.frame_idx),

                # what the app expects:
                "cpu_load": cpu,         # <- expected by app
                "temperature": temp,     # <- expected by app

                # keep existing fields too (nice-to-have + compat)
                "radar_connected": bool(radar_ok),
                "last_frame_age_s": age,
                "clients": server.client_count(),
                "single_client_mode": True,

                # aliases for backward/other tools (harmless)
                "cpu_percent": cpu,
                "temperature_c": temp
            }
            js = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(js)))
            self.end_headers()
            self.wfile.write(js)

        def log_message(self, *_):
            return

    srv = HTTPServer((bind_host, port), Handler)
    threading.Thread(target=srv.serve_forever, name="health-http", daemon=True).start()
    log.info(f"Health on http://{bind_host}:{port}/health")

# ──────────────────────────────────────────────────────────────────────────────
# Signals & main
# ──────────────────────────────────────────────────────────────────────────────
def _handle_exit(sig, frm):
    log.info(f"Signal {sig} received, shutting down…")
    _shutdown.set()
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

def _parse_args():
    # Accept your existing unit’s flags so ExecStart keeps working.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--bind", default=DEF_HOST)
    ap.add_argument("--port", type=int, default=DEF_PORT)
    ap.add_argument("--health-port", type=int, default=DEF_HEALTH_PORT)
    ap.add_argument("--hz", type=float, default=20.0)  # used for rate-limiting sends
    ap.add_argument("--mode", choices=["full","compact"], default="full")  # accepted
    ap.add_argument("--include-heatmaps", dest="send_heatmaps", action="store_true",
                    default=DEF_SEND_HEATMAPS)
    ap.add_argument("--targets-only", action="store_true",
                    default=DEF_TARGETS_ONLY)
    ap.add_argument("--max-pc-points", type=int, default=DEF_MAX_PC_POINTS)
    ap.add_argument("--single-client", action="store_true", default=DEF_SINGLE_CLIENT)  # accepted
    # Parse but ignore other unknown args safely
    args, _ = ap.parse_known_args()
    return args

def main():
    args = _parse_args()

    start_resource_sampler()

    radar = RadarThread(
        targets_only=args.targets_only,
        send_heatmaps=args.send_heatmaps,
        max_pc_points=args.max_pc_points
    ); radar.start()

    srv = SingleClientServer(radar, args.bind, args.port, args.hz); srv.start()

    start_health_http(args.bind, args.health_port, radar, srv)

    try:
        while not _shutdown.is_set():
            time.sleep(0.25)
    finally:
        _shutdown.set(); log.info("Exiting.")

if __name__ == "__main__":
    main()

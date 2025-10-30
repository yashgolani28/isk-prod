#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IWR6843ISK Radar Detection Core — Async Pipeline
- Main loop: parse radar → classify → track → enqueue work only
- Workers: violation (snapshot + annotate), DB insert, heatmap saver
- Detailed logging throughout + periodic health metrics
"""

import os
import sys
import time
import json
import csv
import atexit
import signal
import argparse
import traceback
from datetime import datetime
from collections import deque, defaultdict
from threading import Thread, Lock, Event
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple, Iterable
import numpy as np
import cv2
import matplotlib.pyplot as plt
import psycopg2
from psycopg2 import sql
import math
from pathlib import Path
import subprocess
import base64
import uuid, hashlib, hmac, secrets

try:
    # OpenCV 4.x
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    try:
        # Fallback (older OpenCV)
        cv2.setLogLevel(0)
    except Exception:
        pass

# Local modules
from kalman_filter_tracking import ObjectTracker
from classify_objects import ObjectClassifier
from bounding_box import annotate_speeding_object
from anpr import run_anpr
from camera import (
    capture_snapshot,
    init_camera_pipeline,
    request_violation_clip_multi,
    get_ring_stats_multi,
    shutdown_video_pipeline,
)
try:
    from ptz import PTZController, PTZError
except Exception:
    PTZController = None  # PTZ remains optional
    PTZError = RuntimeError
from config_utils import load_config, db_dsn
from kafka_bus import KafkaBus
from radar_logger import IWR6843Logger
from logger import logger
from plotter import Live3DPlotter
try:
    import socketio as _socketio  # optional: python-socketio client
except Exception:
    _socketio = None
from projection import CameraModel, try_load_active_model, radar_obj_to_ptz_angles
from calibration import MODEL_PATH as _CAM_MODEL_PATH, add_pair

# ──────────────────────────────────────────────────────────────────────────────
# Radar source selector (config → env fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _radar_interface_factory(cfg: dict):
    """
    Decide between network client (PC) and UART (Pi) based on config['radar_over_ip'].
    Env override: RADAR_SOURCE=pc|pi still works.
    Returns (RadarInterfaceClass, kwargs)
    """
    role = str((cfg.get("radar_over_ip") or {}).get("role", "pc")).lower()
    enabled = bool((cfg.get("radar_over_ip") or {}).get("enabled", False))
    env_src = os.getenv("RADAR_SOURCE", "").lower()
    use_pc = (env_src == "pc") or (enabled and role == "pc")
    if use_pc:
        # Network client talking to Pi bridge
        from pc_radar_client import IWR6843Interface as _RI
        rip = (cfg.get("radar_over_ip") or {})
        host = os.getenv("PI_HOST", str(rip.get("host") or "192.168.1.108"))
        port = int(os.getenv("PI_PORT",  rip.get("port") or 55000))
        return _RI, {"host": host, "port": port}
    else:
        # Legacy local UART
        from iwr6843_interface import IWR6843Interface as _RI
        return _RI, {}

# ──────────────────────────────────────────────────────────────────────────────
# DB Connection (env → config)
# ──────────────────────────────────────────────────────────────────────────────
def _db_connect():
    """
    Priority:
      1) env DB_DSN or DATABASE_URL
      2) config["db"] composed DSN
    """
    dsn = os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        # config is loaded right below; reuse it to build DSN
        try:
            dsn = db_dsn(config)
        except Exception:
            dsn = "dbname=iwr6843_db user=radar_user host=localhost sslmode=disable"
    return psycopg2.connect(dsn)

# ──────────────────────────────────────────────────────────────────────────────
# Config / Globals
# ──────────────────────────────────────────────────────────────────────────────
config = load_config()
kafka_bus = None
try:
    kafka_bus = KafkaBus(config)
except Exception:
    kafka_bus = None
ABS_SPEED_CAP_KMH = float(config.get("absolute_max_speed_kmh", 220.0))   
ABS_DIST_CAP_M    = float(config.get("absolute_max_distance_m", 200.0)) 
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT") 
# RF constants for doppler→velocity conversion
try:
    _fc_ghz = float(config.get("radar", {}).get("center_frequency_ghz", 60.0))
except Exception:
    _fc_ghz = 60.0
_LAMBDA_M = 299_792_458.0 / (_fc_ghz * 1e9)
_DOP_MIN_HZ = float(config.get("doppler_min_hz", 0.0))

# ──────────────────────────────────────────────────────────────────────────────
# Selected camera helpers (for per-camera calibration/model)
# ──────────────────────────────────────────────────────────────────────────────
def _selected_camera_id() -> str:
    """
    Resolve the camera ID we consider 'selected' for calibration/model:
    - If config['selected_camera'] is a valid index → that camera's id/name
    - Else fall back to role='primary'
    - Else config['primary_camera_id'] or 'primary'
    """
    cams = config.get("cameras") or []
    if isinstance(cams, dict):
        cams = [cams]
    # explicit index from UI
    try:
        idx = config.get("selected_camera", None)
        if isinstance(idx, (int, float)):
            i = int(idx)
            if 0 <= i < len(cams):
                cam = cams[i]
                return str(cam.get("id") or cam.get("name") or "primary")
    except Exception:
        pass
    # fall back to role=primary
    for c in cams:
        if str(c.get("role", "")).lower() == "primary":
            return str(c.get("id") or c.get("name") or "primary")
    # final fallback
    return str(config.get("primary_camera_id") or "primary")

# Initial model load & mtime for hot-reload
try:
    # Prefer the per-selected-camera model; fall back to legacy global.
    _CAM_MODEL = try_load_active_model(_selected_camera_id())
    if _CAM_MODEL is None and os.path.exists(_CAM_MODEL_PATH):
        _CAM_MODEL = CameraModel.load(_CAM_MODEL_PATH)
    _CAM_MODEL_MTIME = os.path.getmtime(_CAM_MODEL_PATH) if os.path.exists(_CAM_MODEL_PATH) else None
except Exception:
    _CAM_MODEL, _CAM_MODEL_MTIME = None, None

# Buffers / Queues
violation_state = {}       # oid -> {"active": bool, "last_change": float}
last_trigger_time = {}
from collections import defaultdict
last_calib_shot = defaultdict(float)  # per-track cooldown for calibration snapshots   # oid -> last enqueue time
frame_buffer = deque(maxlen=6)                         # local frame cache (sharpness-based)
speeding_buffer = defaultdict(lambda: deque(maxlen=5)) # persistence buffer
acceleration_cache = defaultdict(lambda: deque(maxlen=5))
_speed_median = defaultdict(lambda: deque(maxlen=3))
last_snapshot_ids = {}

# ──────────────────────────────────────────────────────────────────────────────
# Display-ID allocator (DB-backed, per-type per-day monotonic counters)
# ──────────────────────────────────────────────────────────────────────────────
_TRK_TO_DISPLAY = {}         
_LAST_SEEN_TRK = {}          
TRACK_TTL_SECONDS = 2.0       

def _sanitize_type(t: str) -> str:
    t = (t or "unknown").strip().lower()
    return "".join(ch for ch in t if ch.isalnum() or ch in ("-", "_")) or "unknown"

def _ensure_id_counter_table():
    try:
        with _db_connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS object_id_counters (
                    date_key   VARCHAR(8)  NOT NULL,
                    obj_type   VARCHAR(64) NOT NULL,
                    last_num   INTEGER     NOT NULL,
                    PRIMARY KEY(date_key, obj_type)
                );
            """)
            conn.commit()
    except Exception as e:
        logger.warning(f"[DISPLAY-ID] Table ensure failed (will retry on next call): {e}")

def _allocate_display_id(obj_type: str, ts: float) -> str:
    """Atomically increments and returns next display ID for (date_key, obj_type)."""
    date_key = datetime.fromtimestamp(ts).strftime("%Y%m%d")
    typ = _sanitize_type(obj_type)
    try:
        with _db_connect() as conn:
            cur = conn.cursor()
            # Insert row if absent
            cur.execute("""
                INSERT INTO object_id_counters (date_key, obj_type, last_num)
                VALUES (%s, %s, 0)
                ON CONFLICT (date_key, obj_type) DO NOTHING;
            """, (date_key, typ))
            # Bump and fetch in one round-trip
            cur.execute("""
                UPDATE object_id_counters
                   SET last_num = last_num + 1
                 WHERE date_key = %s AND obj_type = %s
             RETURNING last_num;
            """, (date_key, typ))
            n = cur.fetchone()[0]
            conn.commit()
            return f"{typ}_{date_key}_{int(n)}"
    except Exception as e:
        logger.error(f"[DISPLAY-ID] Allocation failed; falling back to ephemeral: {e}")
    # Safe fallback (process-unique) if DB unavailable
    n = int(time.time() * 1000) % 1000000
    return f"{typ}_{date_key}_{n}"

# Async job queues
_violation_q = Queue(maxsize=64)   # snapshot + annotation jobs
_db_q        = Queue(maxsize=256)  # DB write jobs
_heatmap_q   = Queue(maxsize=8)    # heatmap save/update jobs (coalesced)
_ptz_q       = Queue(maxsize=64)   # async PTZ commands (fire-and-forget)
_aux_snap_q  = Queue(maxsize=128)  # aux evidence snapshot jobs

# ──────────────────────────────────────────────────────────────────────────────
# PTZ auto-tracking (radar-based)
# - State file toggled by UI: /run/iwr6843isk/ptz_state.json
# - Strategy: if lock_tid set, follow it; else follow newest entrant in FOV
# - Uses azimuth/elevation from radar tracks → continuous nudge commands
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RUNTIME_DIR = os.environ.get("IWR_RUNTIME_DIR", os.path.join(BASE_DIR, "runtime"))
try:
    os.makedirs(RUNTIME_DIR, exist_ok=True)
except Exception:
    pass
PTZ_STATE_PATH  = os.path.join(RUNTIME_DIR, "ptz_state.json")
PTZ_STATUS_PATH = os.path.join(RUNTIME_DIR, "ptz_status.json")
try:
    os.makedirs(os.path.dirname(PTZ_STATE_PATH), exist_ok=True)
except Exception:
    pass

_ptz_state = {"enabled": False, "lock_tid": None, "last_load": 0.0}
_ptz_seen_first: dict[str, float] = {}   # tid → first-seen time
_ptz_last_cmd = 0.0
_ptz_prev_enabled = False 
_ptz_manual_pause_until = 0.0
_ptz_last_target_tid = ""

def _ptz_cfg_val(key: str, default):
    try:
        return (_ptz_cfg().get(key, default))
    except Exception:
        return default

def _ptz_pause(seconds: float):
    """Extend the auto-follow pause window by `seconds`."""
    global _ptz_manual_pause_until
    try:
        seconds = float(seconds)
    except Exception:
        seconds = 0.0
    _ptz_manual_pause_until = max(_ptz_manual_pause_until, time.time() + max(0.0, seconds))

def _ptz_cfg() -> dict:
    """
    Resolve PTZ config from multiple places:
      1) config["ptz"] if it has host/ip
      2) a camera with role="ptz" in config["cameras"] (prefer nested cam["ptz"])
      3) any camera exposing PTZ fields at top level
    Normalizes "ip" → "host" for downstream users.
    """
    try:
        # 1) explicit top-level
        ptz = dict(config.get("ptz") or {})
        if ptz.get("host") or ptz.get("ip"):
            if "host" not in ptz and ptz.get("ip"):
                ptz["host"] = ptz["ip"]
            return ptz

        # cameras could be list or single dict
        cams = config.get("cameras") or []
        if isinstance(cams, dict):
            cams = [cams]

        # 2) prefer role='ptz'
        for c in cams:
            if str(c.get("role", "")).lower() == "ptz":
                cfg = dict(c.get("ptz") or {})
                # overlay top-level camera creds if present
                for k in ("host", "ip", "username", "password", "port", "profile_token"):
                    if k in c and c[k] and k not in cfg:
                        cfg[k] = c[k]
                if cfg.get("ip") and not cfg.get("host"):
                    cfg["host"] = cfg["ip"]
                if cfg.get("host"):
                    return cfg

        # 3) any camera that exposes PTZ fields
        for c in cams:
            if isinstance(c.get("ptz"), dict):
                cfg = dict(c["ptz"])
                if cfg.get("ip") and not cfg.get("host"):
                    cfg["host"] = cfg["ip"]
                if cfg.get("host"):
                    return cfg
            if c.get("host") or c.get("ip"):
                return {
                    "host": c.get("host") or c.get("ip"),
                    "username": c.get("username"),
                    "password": c.get("password"),
                    "port": c.get("port") or 80,
                    "profile_token": c.get("profile_token", "")
                }
        return {}
    except Exception:
        return {}

# Tunables (can be overridden by config["ptz"])
_ptz_cmd_rate_hz   = float(_ptz_cfg().get("max_rate_hz",   3.0))   # max nudge frequency
_ptz_deadband_deg  = float(_ptz_cfg().get("deadband_deg",  1.5))   # no move inside this error
_ptz_kp            = float(_ptz_cfg().get("kp",            0.015)) # velocity per degree
_ptz_nudge_sec     = float(_ptz_cfg().get("nudge_seconds", 0.35))

# Orientation offsets between radar frame and PTZ optical axis (deg)
_ptz_yaw_offset_deg  = float(_ptz_cfg().get("yaw_offset_deg",  0.0))
_ptz_tilt_offset_deg = float(_ptz_cfg().get("tilt_offset_deg", 0.0))

# Only allow zoom when pointing error is small (deg)
_ptz_zoom_enable_err_deg = float(_ptz_cfg().get("zoom_enable_error_deg", 4.0))

#  low-pass on distance per TID to de-jitter zoom
_PTZ_DIST_LP = {}  # tid -> smoothed distance
_PTZ_DIST_ALPHA = float(_ptz_cfg().get("zoom_distance_alpha", 0.45))

# --- Zoom tuning (all overrideable via config["ptz"]) ---
# Keep the target roughly at this distance; positive error → zoom in, negative → zoom out.
_ptz_zoom_target_m   = float(_ptz_cfg().get("zoom_target_m",   12.0))
_ptz_zoom_deadband_m = float(_ptz_cfg().get("zoom_deadband_m",  2.0))
_ptz_zoom_kp         = float(_ptz_cfg().get("zoom_kp",          0.08))  # normalized zoom per meter
_ptz_zoom_seconds    = float(_ptz_cfg().get("zoom_seconds",     _ptz_nudge_sec))

# --- Mount & Home config (geometry + startup pose) ---
def _ptz_mount_cfg() -> dict:
    """Return PTZ mount dict; defaults to dz=-1m (camera ~1m below radar)."""
    ptz = _ptz_cfg()
    m = dict(ptz.get("mount") or {})
    # sensible defaults if not set
    m.setdefault("dx_m", float(ptz.get("dx_m", 0.0)))
    m.setdefault("dy_m", float(ptz.get("dy_m", 0.0)))
    m.setdefault("dz_m", float(ptz.get("dz_m", -1.0)))
    m.setdefault("yaw_deg",  float(ptz.get("yaw_offset_deg",  0.0)))
    m.setdefault("pitch_deg", float(ptz.get("pitch_offset_deg", 0.0)))
    m.setdefault("roll_deg",  float(ptz.get("roll_offset_deg",  0.0)))
    return m

def _ptz_home_cfg() -> dict:
    """Home pose to start tracking from."""
    ptz = _ptz_cfg()
    home = dict(ptz.get("home") or {})
    # allow legacy single fields
    home.setdefault("pan_deg",  float(ptz.get("home_pan_deg",   0.0)))
    home.setdefault("tilt_deg", float(ptz.get("home_tilt_deg", -5.0)))
    home.setdefault("zoom",     float(ptz.get("home_zoom",      0.0)))
    # optional preset name/id
    if "preset" not in home:
        if ptz.get("home_preset") or ptz.get("preset"):
            home["preset"] = ptz.get("home_preset") or ptz.get("preset")
    # behavior
    home.setdefault("on_enable", bool(ptz.get("home_on_enable", True)))
    home.setdefault("settle_s",  float(ptz.get("home_settle_s", 0.6)))
    home.setdefault("wait_settle", bool(ptz.get("home_wait_settle", True)))
    home.setdefault("settle_samples",     int(ptz.get("settle_samples", 2)))
    home.setdefault("settle_sample_sleep_s", float(ptz.get("settle_sample_sleep_s", 0.15)))
    return home

def _ptz_load_state():
    """Refresh UI state from PTZ_STATE_PATH if modified."""
    try:
        m = os.path.getmtime(PTZ_STATE_PATH)
        if m <= _ptz_state["last_load"]:
            return
        with open(PTZ_STATE_PATH, "r") as f:
            data = json.load(f) or {}
        _ptz_state["enabled"]   = bool(data.get("enabled", False))
        _ptz_state["lock_tid"]  = data.get("lock_tid")
        _ptz_state["last_load"] = m
    except Exception:
        # keep last values on any error
        pass

def _ptz_write_status(enabled: bool, locked_by: str | None):
    """Publish a small status JSON so the web UI can show the correct warning."""
    try:
        os.makedirs(os.path.dirname(PTZ_STATUS_PATH), exist_ok=True)
        payload = {
            "enabled": bool(enabled),
            "locked_by": (locked_by or "") if enabled else "",
            "updated_at": time.time()
        }
        tmp = PTZ_STATUS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, PTZ_STATUS_PATH)
    except Exception:
        pass

def _ptz_apply_state():
    """
    Edge-detect the UI toggle and (un)lock PTZ via worker.
    This runs every frame, independent of targets, so lock happens immediately.
    """
    global _ptz_prev_enabled
    _ptz_load_state()
    cur = bool(_ptz_state.get("enabled", False))
    if cur != _ptz_prev_enabled:
        try:
            if cur:
                # Acquire PTZ for auto tracking
                _ptz_q.put_nowait({"type": "lock", "owner": "auto"})
                _ptz_write_status(True, "auto")
                logger.info("[PTZ] Auto-track enabled -> PTZ locked by 'auto'")
                _ptz_q.put_nowait({"type": "stop"})
                _ptz_pause(_ptz_cfg_val("pause_after_toggle_s", 0.8))
                home = _ptz_home_cfg()
                if home.get("on_enable", True):
                    # Prefer preset if provided; else absolute pan/tilt/zoom.
                    if "preset" in home and str(home["preset"]).strip():
                        _ptz_q.put_nowait({"type": "preset", "preset": str(home["preset"]), "owner": "auto"})
                        # Pose estimate is unknown on presets; let UI optionally provide home pan/tilt as well.
                        if "pan_deg" in home and "tilt_deg" in home:
                            _ptz_q.put_nowait({"type": "pose_estimate",
                                               "pan": float(home["pan_deg"]), "tilt": float(home["tilt_deg"])})
                    else:
                        _ptz_q.put_nowait({"type": "abs",
                                           "pan": float(home.get("pan_deg", 0.0)),
                                           "tilt": float(home.get("tilt_deg", -5.0)),
                                           "zoom": float(home.get("zoom", 0.0)),
                                           "owner": "auto"})
                        _ptz_q.put_nowait({"type": "pose_estimate",
                                           "pan": float(home.get("pan_deg", 0.0)),
                                           "tilt": float(home.get("tilt_deg", -5.0))})
                    _ptz_pause(float(home.get("settle_s", 0.6)))
                    if home.get("wait_settle", True):
                        _ptz_q.put_nowait({
                            "type": "settle",
                            "timeout_s": float(home.get("settle_s", 0.6)),
                            "samples":    int(home.get("settle_samples", 2)),
                            "sleep_s":    float(home.get("settle_sample_sleep_s", 0.15)),
                        })
                    else:
                        _ptz_pause(float(home.get("settle_s", 0.6)))
            else:
                # Release PTZ to allow manual control
                _ptz_q.put_nowait({"type": "unlock", "owner": "auto"})
                _ptz_write_status(False, None)
                logger.info("[PTZ] Auto-track disabled -> PTZ unlocked")
                _ptz_q.put_nowait({"type": "stop"})
                _ptz_pause(_ptz_cfg_val("pause_after_toggle_s", 0.8))
        except Exception:
            logger.debug("[PTZ] state apply failed", exc_info=True)
        _ptz_prev_enabled = cur

def _choose_ptz_target(tracked: list[dict]) -> dict | None:
    """Pick UI-locked TID; otherwise newest first-seen live track."""
    now = time.time()
    for o in tracked or []:
        tid = str(o.get("track_id") or o.get("tid") or o.get("object_id") or "")
        if tid and tid not in _ptz_seen_first:
            _ptz_seen_first[tid] = now
    # Prefer UI lock
    lock = _ptz_state.get("lock_tid")
    if lock:
        for o in tracked or []:
            tid = str(o.get("track_id") or o.get("tid") or o.get("object_id") or "")
            if tid == lock:
                return o
   # Else newest entrant
    best, best_t = None, -1.0
    for o in tracked or []:
        tid = str(o.get("track_id") or o.get("tid") or o.get("object_id") or "")
        t0  = _ptz_seen_first.get(tid, 0.0)
        if t0 > best_t:
            best, best_t = o, t0
    return best

def _ptz_auto_follow(tracked: list[dict]):
    """Rate-limited auto-follow using radar az/el; enqueues nudge to PTZ worker."""
    if not (_ptz_cfg().get("host")):
        return
    _ptz_apply_state()
    if not _ptz_state.get("enabled", False):
        return
    if time.time() < _ptz_manual_pause_until:
        return
    global _ptz_last_cmd
    now = time.time()
    if (now - _ptz_last_cmd) < (1.0 / max(0.5, _ptz_cmd_rate_hz)):
        return
    tgt = _choose_ptz_target(tracked)
    if not tgt:
        return
    # derive az/el if missing
    try:
        az = float(tgt.get("azimuth",  math.degrees(math.atan2(float(tgt.get("x",0.0)), max(1e-6,float(tgt.get("y",0.0)))))))
        el = float(tgt.get("elevation",math.degrees(math.atan2(float(tgt.get("z",0.0)), max(1e-6, (float(tgt.get("x",0.0))**2 + float(tgt.get("y",0.0))**2) ** 0.5)))))
    except Exception:
        return
    # distance for zoom control
    try:
        dist = float(tgt.get("distance"))
    except Exception:
        try:
            x = float(tgt.get("x", 0.0)); y = float(tgt.get("y", 0.0)); z = float(tgt.get("z", 0.0))
            dist = (x*x + y*y + z*z) ** 0.5
        except Exception:
            dist = None

    try:
        pan_deg, tilt_deg = radar_obj_to_ptz_angles(
            r_m=max(0.05, float(dist) if dist is not None else float(tgt.get("range", 0.0) or 0.0)),
            az_deg=float(az), el_deg=float(el),
            mount=_ptz_mount_cfg()
        )
    except Exception:
        # fallback: treat radar az/el as if they were PTZ errors
        pan_deg, tilt_deg = float(az), float(el)

    # First-time / target change log
    try:
        tid = str(tgt.get("track_id") or tgt.get("tid") or "")
        global _ptz_last_target_tid
        if tid and tid != _ptz_last_target_tid:
            logger.info(f"[PTZ] target -> {tid} | pan={pan_deg:.1f}° tilt={tilt_deg:.1f}° dist={(dist if dist is not None else -1):.1f} m")
            _ptz_last_target_tid = tid
    except Exception:
        pass
    # Enqueue a driftless tracking step (PTZ worker will call auto_track_step)
    try:
        _ptz_q.put_nowait({
            "type": "track_angles",
            "pan": float(pan_deg),
            "tilt": float(tilt_deg),
            "dist": (None if dist is None else float(dist)),
            "owner": "auto"
       })
        _ptz_last_cmd = now
    except Exception:
        pass

# Metrics
_METRICS = {
    "enq_violation": 0, "done_violation": 0, "drop_violation": 0,
    "enq_db": 0,        "done_db": 0,        "drop_db": 0,
    "enq_heatmap": 0,   "done_heatmap": 0,   "drop_heatmap": 0,
    "frames": 0,        "classified": 0,     "tracked": 0
}
_METRICS_LOCK = Lock()

# Paths
violations_csv = "radar-logs/violations.csv"
os.makedirs("radar-logs", exist_ok=True)
os.makedirs("system-logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Unified per-violation bundle helpers
# ──────────────────────────────────────────────────────────────────────────────
BUNDLES_DIR = "violations"
os.makedirs(BUNDLES_DIR, exist_ok=True)

def _ensure_web_playable_mp4(path: str) -> None:
    """
    Idempotently ensure MP4 has the moov atom at the head (browser-seekable).
    Uses: ffmpeg -c copy -movflags +faststart (remux in place).
    Safe to call repeatedly; no-op if already faststart.
    """
    try:
        if not path or not path.lower().endswith(".mp4") or not os.path.isfile(path):
            return
        # Quick heuristic: if 'moov' comes before 'mdat' in first ~1MB, likely already OK.
        try:
            with open(path, "rb") as f:
                head = f.read(1048576)
            moov = head.find(b"moov")
            mdat = head.find(b"mdat")
            if moov != -1 and (mdat == -1 or moov < mdat):
                return
        except Exception:
            pass
        tmp = path + ".fs.tmp"
        cmd = ["ffmpeg", "-y", "-i", path, "-c", "copy", "-movflags", "+faststart", tmp]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
        else:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"[MP4] faststart skipped for {path}: {e}")

def _bundle_dir(prefix: str, oid: str, ts: str) -> str:
    """
    ts format: 'YYYYMMDD_HHMMSS_micro' (we generate this near annotation).
    Folder: violations/YYYYMMDD/<prefix>_<oid>_<ts>/
    """
    date_key = ts.split('_')[0]
    return os.path.join(BUNDLES_DIR, date_key, f"{prefix}_{oid}_{ts}")

def _ensure_symlink(link_path: str, target_path: str, rel_start: str):
    """Create/replace a relative symlink for backward-compatible paths."""
    try:
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        if os.path.lexists(link_path):
            os.remove(link_path)
        rel = os.path.relpath(target_path, start=rel_start)
        os.symlink(rel, link_path)
    except Exception as e:
        logger.debug(f"[BUNDLE] symlink failed {link_path} -> {target_path}: {e}")

def _read_clip_manifest_sidecar(clip_path: str) -> dict:
    """Read JSON sidecar if present: returns {} on failure."""
    try:
        mpath = os.path.splitext(clip_path)[0] + ".json"
        if os.path.exists(mpath):
            with open(mpath, "r") as f:
                man = json.load(f)
            return man.get("meta", man) or {}
    except Exception:
        pass
    return {}

def _move_clip_into_bundle(src: str, bundle_dir: str) -> Optional[str]:
    """Ensure faststart, move into bundle as clip(.mp4|_NN.mp4), add symlink back, update meta."""
    try:
        if not (src and os.path.isfile(src) and src.lower().endswith(".mp4")):
            return None
        try:
           _ensure_web_playable_mp4(src)
        except Exception as _e:
            logger.debug(f"[BUNDLE] faststart pre-move skipped: {_e}")
        base = "clip.mp4"
        dst = os.path.join(bundle_dir, base)
        if os.path.exists(dst):
            i = 2
            while True:
                base = f"clip_{i:02d}.mp4"
                dst = os.path.join(bundle_dir, base)
                if not os.path.exists(dst):
                    break
                i += 1
        os.replace(src, dst)
        # Kafka: publish clip saved event with metadata
        try:
            if kafka_bus and bool(config.get("kafka", {}).get("enabled")):
                kafka_bus.publish(
                    topic=str(config.get("kafka", {}).get("topic_clips", "camera_clips")),
                    value={
                        "event": "clip_saved",
                        "bundle": os.path.basename(bundle_dir),
                        "clip": os.path.basename(dst),
                        "path": os.path.abspath(dst),
                        "ts": time.time(),
                    },
                    key=os.path.basename(bundle_dir),
                )
        except Exception:
            pass
        _ensure_symlink(src, dst, rel_start=os.path.dirname(src) or ".")
        # update meta.json (append and keep 'clip' if absent)
        try:
            mpath = os.path.join(bundle_dir, "meta.json")
            if os.path.exists(mpath):
                with open(mpath, "r") as f:
                    m = json.load(f)
            else:
                m = {}
            clips = list(m.get("clips", []))
            clips.append(os.path.basename(dst))
            seen = set(); clips = [c for c in clips if not (c in seen or seen.add(c))]
            m["clips"] = clips
            if "clip" not in m:
                m["clip"] = os.path.basename(dst)
            with open(mpath, "w") as f:
                json.dump(m, f, indent=2)
        except Exception as _e:
            logger.debug(f"[BUNDLE] meta clips update skipped: {_e}")
        return dst
    except Exception as e:
        logger.debug(f"[BUNDLE] move clip failed: {e}")
        return None

def _absorb_all_matching_clips(bundle_dir: str, oid: str, when_dt: datetime, window_s: float = 8.0) -> List[str]:
    """
    Move *all* matching clips from 'clips/' into bundle:
      - name startswith 'violation_YYYYMMDD_HHMMSS'
      - OR contains the object_id
      - OR sidecar.manifest has matching object_id and time within ±window_s
    Returns list of moved absolute paths (may be empty).
    """
    moved: List[str] = []
    try:
        clips_dir = "clips"
        os.makedirs(clips_dir, exist_ok=True)
        date_str = when_dt.strftime("%Y%m%d")
        time_str = when_dt.strftime("%H%M%S")
        prefix = f"violation_{date_str}_{time_str}"
        for name in list(os.listdir(clips_dir)):
            if not name.lower().endswith(".mp4"):
                continue
            path = os.path.join(clips_dir, name)
            if not os.path.isfile(path):
                continue
            ok = False
            # rule 1: exact timestamp prefix
            if name.startswith(prefix):
                ok = True
            # rule 2: strict sidecar match → SAME object_id AND within a tight time window
            if not ok:
                meta = _read_clip_manifest_sidecar(path)
                mid  = str(meta.get("object_id") or "")
                et   = meta.get("event_time")
                try:
                    mdt = datetime.fromisoformat(et) if et else None
                except Exception:
                   mdt = None
                if (oid and mid and (oid == mid)) and mdt:
                    if abs((mdt - when_dt).total_seconds()) <= window_s:
                        ok = True
            if not ok:
                continue
            dst = _move_clip_into_bundle(path, bundle_dir)
            if dst:
                moved.append(dst)
    except Exception as e:
        logger.debug(f"[BUNDLE] absorb-all failed: {e}")
    return moved

def _absorb_clip_into_bundle_now(bundle_dir: str, oid: str, ts: str) -> Optional[str]:
    """
    Back-compat shim: absorb *all* matches, return the first moved (if any).
    """
    when = _parse_bundle_ts(ts) or datetime.now()
    moved = _absorb_all_matching_clips(bundle_dir, oid, when)
    return moved[0] if moved else None

def _write_meta(bundle_dir: str, obj: dict, image_name: str,
                clip_name: str | None, camera: dict | None, frame_meta: dict | None):
    meta = {
        "created_at": datetime.now().isoformat(),
        "object": {k: to_native(v) for k, v in (obj or {}).items()},
        "image": image_name,
        "clip": clip_name,
        "camera": camera or {},
        "frame_meta": frame_meta or {},
    }
    try:
        with open(os.path.join(bundle_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.debug(f"[BUNDLE] write meta failed: {e}")

def _parse_bundle_ts(ts_str: str):
    # 'YYYYMMDD_HHMMSS_micro' → datetime
    try:
        # micro may be missing (defensive), so split and pad
        if ts_str.count('_') == 2:
            # YYYYMMDD_HHMMSS_micro
            ymd, hms, micro = ts_str.split('_', 2)
            base = datetime.strptime(f"{ymd}_{hms}", "%Y%m%d_%H%M%S")
            return base.replace(microsecond=int(micro))
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception:
        return None

def _find_bundle_for(obj_id: str, when: datetime, prefix: str = "violation"):
    """
    Locate an existing bundle for (obj_id, around 'when') within ±5 min.
    """
    try:
        day_dir = os.path.join(BUNDLES_DIR, when.strftime("%Y%m%d"))
        if not os.path.isdir(day_dir):
            return None
        best = None
        best_dt = None
        for name in os.listdir(day_dir):
            if not name.startswith(f"{prefix}_{obj_id}_"):
                continue
            ts_str = name.split(f"{prefix}_{obj_id}_", 1)[-1]
            bdt = _parse_bundle_ts(ts_str)
            if bdt is None:
                continue
            # keep the nearest within ±5 minutes
            if abs((bdt - when).total_seconds()) <= 300:
                if best is None or abs((bdt - when).total_seconds()) < abs((best_dt - when).total_seconds()):
                    best = os.path.join(day_dir, name)
                    best_dt = bdt
        return best
    except Exception as e:
        logger.debug(f"[BUNDLE] find failed: {e}")
        return None

# CSV logger
radar_csv_logger = IWR6843Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def to_native(val):
    if isinstance(val, (np.generic, np.ndarray)):
        try:
            return val.item()
        except Exception:
            return float(val)
    return val

def _safe_speed_kmh(x, hi: float | None = None) -> float | None:
    """
    Return a clean km/h or None if non-finite/implausible.
    hi defaults to ABS_SPEED_CAP_KMH.
    """
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        v = abs(v)
        ceiling = float(hi if hi is not None else ABS_SPEED_CAP_KMH)
        if v < 0.0 or v > ceiling:
            return None
        return v
    except Exception:
        return None

def _sanitize_obj_physics(obj: dict) -> None:
    """
    Clamp/clean *in place*:
      - speed_kmh / velocity
      - distance
    No raises; ensures finite values before UI/DB.
    """
    # speed/velocity
    v_kmh = _safe_speed_kmh(obj.get("speed_kmh", 0.0))
    if v_kmh is None:
        obj["speed_kmh"] = 0.0
        obj["velocity"] = 0.0
    else:
        obj["speed_kmh"] = float(v_kmh)
        obj["velocity"]  = float(v_kmh / 3.6)

    # distance
    try:
        d = float(obj.get("distance", obj.get("range", 0.0)))
        if not math.isfinite(d) or d < 0.0:
            d = 0.0
        if d > ABS_DIST_CAP_M:
            d = ABS_DIST_CAP_M
        obj["distance"] = float(d)
    except Exception:
        obj["distance"] = 0.0

def _is_cam_active(cam: dict) -> bool:
    """
    Treat a camera as active if ANY of these are truthy (to support mixed configs):
      - enabled
      - is_active (legacy)
      - active (alias seen in some UIs/JSON)
    Accepts 1/0, true/false, "1"/"true"/"on"/"yes"
    """
    if not cam:
        return False
    # Prefer explicit enabled, then legacy/new aliases
    v = cam.get("enabled", None)
    if v is None:
        v = cam.get("is_active", None)
    if v is None:
        v = cam.get("active", True)
    try:
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "on", "yes")
        if isinstance(v, (int, float)):
            return int(v) == 1
        return bool(v)
    except Exception:
        return bool(v)

def _active_cameras() -> list[dict]:
    cams = config.get("cameras") or []
    if isinstance(cams, dict):
        cams = [cams]
    return [c for c in cams if _is_cam_active(c)]

def _select_primary(cams: list[dict]) -> dict:
    """STRICT: only role='primary' qualifies; otherwise no primary."""
    for c in cams:
        if str(c.get("role", "")).lower() == "primary":
            return c
    return {}

_LAST_PROGRESS_TS = time.monotonic()

def get_targets_with_timeout(radar, timeout_s=0.8):
    """
    Call radar.get_targets() in a tiny worker thread and wait up to timeout_s.
    Returns a dict frame, or None if timed out (i.e., likely stalled read).
    """
    result = {}
    done = Event()

    def _worker():
        nonlocal result
        try:
            result = radar.get_targets()
        except Exception as e:
            logger.warning(f"[RADAR] get_targets error: {e}")
            result = {}
        finally:
            done.set()

    t = Thread(target=_worker, daemon=True)
    t.start()
    finished = done.wait(timeout_s)
    return result if finished else None


def _maybe_hard_exit_if_stalled(seconds_without_progress, limit_s=15):
    if seconds_without_progress >= limit_s:
        logger.error(f"[WATCHDOG] main loop stalled for {seconds_without_progress:.1f}s -> exiting for restart")
        os._exit(21)  # let systemd restart us

def handle_exit(signum, frame):
    logger.info(f"[EXIT] Received signal {signum}, shutting down…")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def _init_primary_video_pipeline():
    """
    Initialize the multi-cam ring+clip worker for the PRIMARY camera stream.
    Only the primary camera gets MP4 clips; aux cams remain snapshot-only.
    """
    try:
        cams = _active_cameras()
        primary = _select_primary(cams)
        if not primary:
            logger.warning("[VIDEO PIPELINE] No primary camera available")
            return
        # prefer RTSP for ring buffer; fall back to MJPEG/snapshot URL
        url = primary.get("rtsp_url") or primary.get("url") or primary.get("snapshot_url")
        if not url:
            logger.warning("[VIDEO PIPELINE] Primary camera has no stream URL")
            return
        init_camera_pipeline(
            cam_id="primary",
            camera_url=url,
            username=primary.get("username"),
            password=primary.get("password"),
            target_fps=float(config.get("clip_fps", 8.0)),
            pre_seconds=float(config.get("clip_pre_seconds", 3.0)),
            post_seconds=float(config.get("clip_post_seconds", 2.0)),
            buffer_margin=float(config.get("clip_buffer_margin", 6.0)),
            timeout=float(config.get("camera_timeout", 3.0)),
            quality=str(config.get("clip_quality", "medium")),
        )
        logger.info(f"[VIDEO PIPELINE] Initialized primary ring for: {primary.get('name','primary')}")
    except Exception:
        logger.exception("[VIDEO PIPELINE] init failed")

atexit.register(shutdown_video_pipeline)

def _camera_id_for(cam: dict, idx: int) -> str:
    """Stable ID for a camera based on config fields."""
    return str(cam.get("id") or cam.get("name") or f"cam{idx+1}")

def _init_video_pipelines_for_all():
    """
    Initialize per-camera ring+clip workers:
      - Always init PRIMARY as cam_id='primary'
      - If aux clips are enabled, init each aux camera with its own cam_id
    """
    try:
        cams = _active_cameras()
        if not cams:
            logger.warning("[VIDEO PIPELINE] No active cameras configured")
            return
        primary = _select_primary(cams)
        # ── primary ──
        p_url = (primary or {}).get("rtsp_url") or (primary or {}).get("url") or (primary or {}).get("snapshot_url")
        if p_url:
            init_camera_pipeline(
                cam_id="primary",
                camera_url=p_url,
                username=primary.get("username"),
                password=primary.get("password"),
                target_fps=float(config.get("clip_fps", 8.0)),
                pre_seconds=float(config.get("clip_pre_seconds", 3.0)),
                post_seconds=float(config.get("clip_post_seconds", 2.0)),
                buffer_margin=float(config.get("clip_buffer_margin", 6.0)),
                timeout=float(config.get("camera_timeout", 3.0)),
                quality=str(config.get("clip_quality", "medium")),
            )
        else:
            logger.warning("[VIDEO PIPELINE] Primary camera missing URL")

        # ── aux (only if enabled) ──
        if bool(config.get("aux_clips_enabled", False)):
            for idx, cam in enumerate(cams):
                if cam is primary:
                    continue
                url = cam.get("rtsp_url") or cam.get("url") or cam.get("snapshot_url")
                if not url:
                    continue
                cam_id = _camera_id_for(cam, idx)
                init_camera_pipeline(
                    cam_id=cam_id,
                    camera_url=url,
                    username=cam.get("username"),
                    password=cam.get("password"),
                    target_fps=float(config.get("clip_fps", 8.0)),
                    pre_seconds=float(config.get("clip_pre_seconds", 3.0)),
                    post_seconds=float(config.get("clip_post_seconds", 2.0)),
                    buffer_margin=float(config.get("clip_buffer_margin", 6.0)),
                    timeout=float(config.get("camera_timeout", 3.0)),
                    quality=str(config.get("clip_quality", "medium")),
                )
    except Exception:
        logger.exception("[VIDEO PIPELINE] init-all failed")

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def roi_motion_score(prev_img, curr_img, bbox):
    """Mean absolute pixel diff inside bbox; static patches are usually <~2–3."""
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
    h, w = curr_img.shape[:2]
    x1 = min(x1, w-1); x2 = min(x2, w); y1 = min(y1, h-1); y2 = min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    p = cv2.cvtColor(prev_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(curr_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    p = cv2.GaussianBlur(p, (5,5), 0)
    c = cv2.GaussianBlur(c, (5,5), 0)
    return float(cv2.absdiff(p, c).mean())

# ──────────────────────────────────────────────────────────────────────────────
# Canonical velocity selection
# ──────────────────────────────────────────────────────────────────────────────
def _apply_canonical_velocity(obj: dict) -> None:
    # (1) Configurable canonical field — removes any derived along-road inflation
    field = str(config.get("violation_speed_field", "speed_radial_kmh"))
    best_mps = None
    try:
        v_kmh_from_field = obj.get(field, None)
        if v_kmh_from_field is not None:
            best_mps = abs(float(v_kmh_from_field)) / 3.6
    except Exception:
        best_mps = None
    # (2) Tracker-reported 3D speed (fallback)
    if best_mps is None:
        try:
            best_mps = float(obj.get("speed_kmh", 0.0)) / 3.6
        except Exception:
            best_mps = None
    # (3) Magnitude from components (last resort)
    if best_mps is None:
        vx = float(obj.get("velX", obj.get("vx", 0.0)) or 0.0)
        vy = float(obj.get("velY", obj.get("vy", 0.0)) or 0.0)
        vz = float(obj.get("velZ", obj.get("vz", 0.0)) or 0.0)
        best_mps = float(np.linalg.norm([vx, vy, vz]))
    # Finalize canonical fields (robust median smoothing)
    tid = obj.get("tid") or obj.get("track_id") or obj.get("object_id") or "UNK"
    _speed_median[tid].append(float(best_mps * 3.6))
    if len(_speed_median[tid]) >= 2:
        med = sorted(_speed_median[tid])[len(_speed_median[tid]) // 2]
    else:
        med = float(best_mps * 3.6)
    obj["velocity"] = float(max(0.0, med / 3.6))
    obj["speed_kmh"] = float(max(0.0, med))

def log_violation_to_csv(obj, note="Snapshot Failed"):
    with open(violations_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.fromtimestamp(obj["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            obj.get("sensor", "IWR6843ISK"),
            obj.get("object_id"),
            obj.get("type"),
            obj.get("confidence", 0.0),
            obj.get("speed_kmh", 0.0),
            obj.get("velocity", 0.0),
            obj.get("distance", 0.0),
            obj.get("direction", "unknown"),
            obj.get("signal_level", 0.0),
            obj.get("doppler_frequency", 0.0),
            obj.get("motion_state", "unknown"),
            note
        ])


def _first_nonempty(frame: dict, keys: tuple[str, ...], default=None):
    """
    Return the first present & non-empty value among `keys` in `frame`,
    without triggering NumPy truthiness errors.
    """
    for k in keys:
        if k not in frame:
            continue
        v = frame[k]
        if v is None:
            continue
        try:
            if isinstance(v, np.ndarray):
                if v.size > 0:
                    return v
            elif isinstance(v, (list, tuple, dict, set)):
                if len(v) > 0:
                    return v
            else:
                # Fallback: accept non-None scalar-like
                return v
        except Exception:
            # Be defensive; skip any weird types
            continue
    return default

def _extract_xyz_doppler(points):
    """
    Accepts either list[dict] (x,y,z,doppler,snr,noise) or list/ndarray with columns:
    [x,y,z,(doppler),(snr),(noise)]. Returns np arrays: X(N,3), D(N,), S(N,), N(N,)
    """
    import numpy as np

    # Robust emptiness checks (avoid "truth value is ambiguous" on ndarray)
    if points is None:
        return (np.empty((0, 3), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32))

    if isinstance(points, np.ndarray):
        if points.size == 0:
            return (np.empty((0, 3), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32))
        A = np.asarray(points, dtype=np.float32)
        X = A[:, :3]
        D = A[:, 3] if A.shape[1] > 3 else np.zeros((A.shape[0],), np.float32)
        S = A[:, 4] if A.shape[1] > 4 else np.zeros((A.shape[0],), np.float32)
        N = A[:, 5] if A.shape[1] > 5 else np.zeros((A.shape[0],), np.float32)
        return X, D, S, N

    if isinstance(points, (list, tuple)):
        if len(points) == 0:
            return (np.empty((0, 3), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32))
        first = points[0]
        if isinstance(first, dict):
            X = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in points], np.float32)
            D = np.array([p.get("doppler", 0.0) for p in points], np.float32)
            S = np.array([p.get("snr", 0.0) for p in points], np.float32)
            N = np.array([p.get("noise", 0.0) for p in points], np.float32)
            return X, D, S, N
        # list/tuple of numerics
        A = np.asarray(points, dtype=np.float32)
        if A.size == 0:
            return (np.empty((0, 3), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32))
        X = A[:, :3]
        D = A[:, 3] if A.shape[1] > 3 else np.zeros((A.shape[0],), np.float32)
        S = A[:, 4] if A.shape[1] > 4 else np.zeros((A.shape[0],), np.float32)
        N = A[:, 5] if A.shape[1] > 5 else np.zeros((A.shape[0],), np.float32)
        return X, D, S, N

    # Fallback for unexpected types
    A = np.asarray(points, dtype=np.float32)
    if A.size == 0:
        return (np.empty((0, 3), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32))
    X = A[:, :3]
    D = A[:, 3] if A.shape[1] > 3 else np.zeros((A.shape[0],), np.float32)
    S = A[:, 4] if A.shape[1] > 4 else np.zeros((A.shape[0],), np.float32)
    N = A[:, 5] if A.shape[1] > 5 else np.zeros((A.shape[0],), np.float32)
    return X, D, S, N

def _cluster_points_dbscan(XYZ: np.ndarray, eps: float = 0.35, min_samples: int = 6) -> np.ndarray:
    """
    Try sklearn.DBSCAN lazily; if unavailable or fails, fall back to a lightweight NumPy radius grouping.
    Returns labels (shape: [n_points], -1 = noise).
    """
    global DBSCAN
    n = int(XYZ.shape[0]) if XYZ is not None else 0
    if n == 0:
        return np.empty((0,), dtype=np.int32)
    # Try sklearn only when actually needed
    try:
        if DBSCAN is None:
            from sklearn.cluster import DBSCAN as _DBSCAN
            DBSCAN = _DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(XYZ)
    except Exception:
        # Pure NumPy fallback: very simple radius clustering (greedy), good enough to form track-like groups.
        labels = -1 * np.ones((n,), dtype=np.int32)
        used = np.zeros((n,), dtype=bool)
        cid = 0
        for i in range(n):
            if used[i]:
                continue
            d = np.linalg.norm(XYZ - XYZ[i], axis=1)
            neigh = np.where(d <= eps)[0]
            if neigh.size >= min_samples:
                labels[neigh] = cid
                used[neigh] = True
                cid += 1
        return labels

def derive_targets_from_pointcloud(points):
    """
    Lightweight DBSCAN-based grouping when trackData is missing.
    Produces 'track-like' dicts expected by the pipeline.
    """
    XYZ, D, S, N = _extract_xyz_doppler(points)
    if XYZ.shape[0] == 0:
        return []
    global DBSCAN
    labels = None
    try:
        if DBSCAN is None:
            from sklearn.cluster import DBSCAN as _DBSCAN
            DBSCAN = _DBSCAN
        labels = _cluster_points_dbscan(XYZ, eps=0.35, min_samples=6)
    except Exception:
        # Pure-NumPy radius clustering fallback (very lightweight).
        # Groups points within ~0.35 m; marks noise as -1.
        try:
            eps = 0.35
            n = XYZ.shape[0]
            labels = -1 * np.ones((n,), dtype=np.int32)
            cid = 0
            used = np.zeros((n,), dtype=bool)
            for i in range(n):
                if used[i]:
                    continue
                d = np.linalg.norm(XYZ - XYZ[i], axis=1)
                neigh = np.where(d <= eps)[0]
                if neigh.size >= 6:
                    labels[neigh] = cid
                    used[neigh] = True
                    cid += 1
        except Exception:
            labels = -1 * np.ones((XYZ.shape[0],), dtype=np.int32)
    targets = []
    cid = 1
    for k in set(labels):
        if k == -1:
            continue
        mask = labels == k
        C = XYZ[mask]
        pos = C.mean(axis=0)
        # Convert average Doppler [Hz] to radial velocity [m/s] using v≈(λ/2)*fd
        d_avg = float(D[mask].mean()) if D.size else 0.0
        has_dop = np.isfinite(d_avg) and (abs(d_avg) >= _DOP_MIN_HZ)
        v_rad = (d_avg * _LAMBDA_M / 2.0) if has_dop else 0.0
        n_pts = int(mask.sum())
        snr_med = float(np.median(S[mask])) if S.size else 0.0
        dop_med = float(np.median(D[mask])) if D.size else 0.0
        dop_std = float(np.std(D[mask])) if D.size else 0.0
        targets.append({
            "pc_id": cid, "source": "pointcloud",
            "posX": float(pos[0]), "posY": float(pos[1]), "posZ": float(pos[2]),
            "velX": 0.0, "velY": v_rad, "velZ": 0.0,
            "snr": snr_med,
            "noise": float(np.median(N[mask])) if N.size else 0.0,
            "speed_kmh": float(abs(v_rad) * 3.6),
            "doppler_frequency": float(d_avg) if has_dop else 0.0,
            "velocity_source": ("doppler_pc" if has_dop else "none"),
            "origin": "pc_cluster",
            "pc_n": n_pts,
            "pc_doppler_med": dop_med,
            "pc_doppler_std": dop_std,
        })
        cid += 1

    if not targets and XYZ.shape[0] >= 1:
        i = int(np.argmax(S) if S.size else 0)
        pos = XYZ[i]
        d = float(D[i]) if D.size else 0.0
        has_dop = np.isfinite(d) and (abs(d) >= _DOP_MIN_HZ)
        v_rad = (d * _LAMBDA_M / 2.0) if has_dop else 0.0
        targets.append({
            "pc_id": 1, "source": "pointcloud",
            "posX": float(pos[0]), "posY": float(pos[1]), "posZ": float(pos[2]),
            "velX": 0.0, "velY": v_rad, "velZ": 0.0,
            "snr": float(S[i]) if S.size else 0.0,
            "noise": float(N[i]) if N.size else 0.0,
            "speed_kmh": float(abs(v_rad) * 3.6),
            "doppler_frequency": float(d) if has_dop else 0.0,
            "velocity_source": ("doppler_pc" if has_dop else "none"),
            "origin": "pc_single",
        })
    return targets

def derive_target_from_heatmap(heatmap, range_res_m=0.043, az_fov_deg=120.0):
    """
    Minimal RA→target fallback when a frame has only heatmaps.
    Returns list with ≤1 target.
    """
    try:
        H = None
        if isinstance(heatmap, np.ndarray):
            H = heatmap
        elif isinstance(heatmap, (list, tuple)):
            arr = np.asarray(heatmap, dtype=np.float32).ravel()
            # try common row counts
            for rows in (32, 48, 64, 96, 128, 240, 256):
                if arr.size % rows == 0:
                    H = arr.reshape(rows, -1)
                    break
        if H is None or H.size == 0:
            return []
        idx = int(np.argmax(H))
        rbin, abin = np.unravel_index(idx, H.shape)
        rows, cols = H.shape
        az = ((abin / max(cols - 1, 1)) - 0.5) * az_fov_deg
        dist = (rbin + 0.5) * range_res_m
        azr = np.deg2rad(az)
        x = dist * np.sin(azr); y = dist * np.cos(azr)
        snr = float(H[rbin, abin])
        return [{
            "source": "heatmap",
            "posX": float(x), "posY": float(y), "posZ": 0.0,
            "velX": 0.0, "velY": 0.0, "velZ": 0.0,
            "snr": snr, "noise": 0.0,
            "speed_kmh": 0.0, "distance": float(dist),
        }]
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Camera Orchestrator: capture ALL cameras, crop/ANPR ONLY on primary
# ──────────────────────────────────────────────────────────────────────────────

class CameraOrchestrator:
    """
    For each violation:
      - Primary camera: enqueue to _violation_q (existing pipeline does bbox, ANPR, clip, bundle).
      - Other cameras: capture burst frames and drop into the SAME bundle as 'aux' images (no crops/ANPR).
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.burst_n = int(cfg.get("burst_frames_per_camera", 3))
        self.burst_window_ms = int(cfg.get("burst_window_ms", 1200))
        self.prefer_ptz_primary = False

    def _active_cameras(self) -> List[Dict]:
        cams = self.cfg.get("cameras") or []
        if isinstance(cams, dict):
            cams = [cams]
        return [c for c in cams if _is_cam_active(c)]

    def _select_primary(self, cams: List[Dict]) -> Optional[Dict]:
        """
        STRICT primary selection:
        - Only a camera with role="primary" is considered primary.
        - If none exists, there is no primary (return None).
        """
        if not cams:
            return None
        for c in cams:
            if str(c.get("role", "")).lower() == "primary":
                return c
        return None

    def _ensure_bundle_for(self, prefix: str, oid: str, ts: str) -> str:
        """Create/get bundle directory using existing convention."""
        bundle_dir = _bundle_dir(prefix, oid, ts)
        os.makedirs(bundle_dir, exist_ok=True)
        return bundle_dir

    def _save_aux_image(self, img_path: str, bundle_dir: str, cam_name: str, stamp: str) -> Optional[str]:
        """Copy/move an already-saved snapshot into bundle as aux image."""
        try:
            # we place aux frames as aux_<camname>_<stamp>.jpg for uniqueness
            base = f"aux_{cam_name}_{stamp}.jpg".replace(" ", "_")
            dst = os.path.join(bundle_dir, base)
            if os.path.abspath(img_path) != os.path.abspath(dst):
                try:
                    os.replace(img_path, dst)
                except Exception:
                    # fallback: copy if replace fails across FS boundaries
                    img = cv2.imread(img_path)
                    if img is not None:
                        cv2.imwrite(dst, img)
            # leave a convenience symlink under snapshots/ (optional)
            try:
                snapshots_root = "snapshots"
                os.makedirs(snapshots_root, exist_ok=True)
                link_path = os.path.join(snapshots_root, base)
                _ensure_symlink(link_path, dst, rel_start=snapshots_root)
            except Exception:
                pass
            return dst
        except Exception as e:
            logger.debug(f"[ORCH] aux image save failed: {e}")
            return None

    def _burst_one_camera(self, cam: Dict, n: int) -> List[str]:
        """Take up to n snapshots from a camera using capture_snapshot(). Returns disk paths."""
        paths = []
        for i in range(max(1, n)):
            try:
                p = capture_snapshot(
                    camera_url=cam.get("snapshot_url") or cam.get("url"),
                    username=cam.get("username"),
                    password=cam.get("password"),
                    timeout=5
                )
                if p and os.path.exists(p):
                    paths.append(p)
            except Exception as e:
                logger.debug(f"[ORCH] snapshot fail on {cam.get('name','cam')} #{i+1}: {e}")
            time.sleep(0.05)
        return paths

    def dispatch(self, obj: dict, heatmap=None, frame_meta=None):
        """Single entrypoint — call this instead of enqueuing directly to _violation_q."""
        cams = self._active_cameras()
        if not cams:
            logger.warning("[ORCH] No active cameras; skipping camera work")
            return

        # Primary camera selection
        primary = self._select_primary(cams)
        if primary is None:
            logger.warning("[ORCH] Primary camera not found; skipping")
            return

        oid = str(obj.get("object_id") or "UNKNOWN")
        try:
            evt_ts = float(obj.get("timestamp", time.time()))
        except Exception:
            evt_ts = time.time()
        when = datetime.fromtimestamp(evt_ts)
        ts = when.strftime("%Y%m%d_%H%M%S_%f")
        prefix = "violation"
        # Try to locate an existing bundle (created by the primary worker) near the event time.
        bundle_dir = _find_bundle_for(oid, when, prefix=prefix)
        if not bundle_dir:
            # If not found yet, create one deterministically from the event time.
            bundle_dir = _bundle_dir(prefix, oid, ts)
            os.makedirs(bundle_dir, exist_ok=True)
        if not bundle_dir:
            # If not found yet, create one deterministically from event time
            bundle_dir = _bundle_dir(prefix, oid, ts)

        # 1) Enqueue the primary camera to existing worker (bbox/ANPR/clip/bundle)
        try:
            job = {
                "obj": dict(obj),         # shallow copy
                "camera": dict(primary),  # contains url/user/pass etc.
                "frame_meta": frame_meta or {},
                "heatmap": heatmap,
                "bundle_ts": ts,
                "camera_id": str(primary.get("id") or primary.get("name") or "primary"),
            }
            _violation_q.put_nowait(job)
            with _METRICS_LOCK:
                _METRICS["enq_violation"] += 1
            logger.info(f"[ORCH] queued primary cam -> {primary.get('name','primary')} for {oid}")
            # Kafka: publish event best-effort
            try:
                if kafka_bus and bool(config.get("kafka", {}).get("enabled")):
                    kafka_bus.publish(
                        topic=str(config.get("kafka", {}).get("topic_violation", "violations")),
                        value={
                            "event": "violation_enqueued",
                            "object_id": oid,
                            "timestamp": evt_ts,
                            "class": str(obj.get("type") or "UNKNOWN"),
                            "speed_kmh": float(obj.get("speed_kmh") or 0.0),
                            "bundle_ts": ts,
                        },
                        key=oid,
                    )
            except Exception:
                pass
        except Exception:
            with _METRICS_LOCK:
                _METRICS["drop_violation"] += 1
            logger.error(f"[ORCH] primary enqueue failed for {oid}")

        # 2) For all *other* cameras: capture burst frames into the SAME bundle (no crops/ANPR)
        for cam in cams:
            if cam is primary:
                continue
            name = str(cam.get("name") or f"cam{cam.get('id','')}")
            burst_paths = self._burst_one_camera(cam, self.burst_n)
            if not burst_paths:
                continue
            for p in burst_paths:
                stamp = os.path.splitext(os.path.basename(p))[0]
                _ = self._save_aux_image(p, bundle_dir, name, stamp)

        # 3) Request clips for primary AND aux (if enabled) so bundle gets all videos.
        try:
            if bool(self.cfg.get("clips_enabled", True)):
                evt_ts = float(obj.get("timestamp", time.time()))
                event_time = datetime.fromtimestamp(evt_ts)
                # primary
                request_violation_clip_multi(
                    "primary",
                    event_id=int(evt_ts * 1000),
                    event_time=event_time,
                    object_id=oid,
                    obj_class=str(obj.get("type") or "UNKNOWN"),
                    speed_kmh=float(obj.get("speed_kmh") or 0.0),
                )
                # aux (if enabled)
                if bool(self.cfg.get("aux_clips_enabled", False)):
                    for idx, cam in enumerate(cams):
                        if cam is primary:
                            continue
                        cam_id = _camera_id_for(cam, idx)
                        try:
                            request_violation_clip_multi(
                                cam_id,
                                event_id=int(evt_ts * 1000),
                                event_time=event_time,
                                object_id=oid,
                                obj_class=str(obj.get("type") or "UNKNOWN"),
                                speed_kmh=float(obj.get("speed_kmh") or 0.0),
                            )
                        except Exception as _e:
                            logger.debug(f"[ORCH] aux clip request skipped for {cam_id}: {_e}")
        except Exception as ce:
            logger.debug(f"[ORCH] clip request skipped: {ce}")


# Orchestrator singleton
_ORCH_SINGLETON: Optional[CameraOrchestrator] = None
def get_orchestrator() -> CameraOrchestrator:
    global _ORCH_SINGLETON
    if _ORCH_SINGLETON is None:
        _ORCH_SINGLETON = CameraOrchestrator(config)
    return _ORCH_SINGLETON

# ──────────────────────────────────────────────────────────────────────────────
# Workers
# ──────────────────────────────────────────────────────────────────────────────
def _violation_worker():
    """
    Consumes violation jobs:
      1) Capture 3 quick snapshots → choose sharpest
      2) Annotate with YOLO
      3) Enqueue DB job (with range/noise profiles and optional heatmap)
    """
    while True:
        job = _violation_q.get()
        try:
            obj = job["obj"]
            cam = job["camera"]
            meta = job.get("frame_meta", {})
            heatmap = job.get("heatmap", None)

            oid = obj.get("object_id", "UNKNOWN")
            logger.info(f"[VIOLATION] Worker start for {oid} | {obj.get('type')} @ {obj.get('speed_kmh',0.0):.1f} km/h")

            # Burst capture: take up to 3 frames and pick the sharpest (thread-local, isolated)
            candidates = []
            for i in range(3):
                try:
                    path = capture_snapshot(
                        camera_url=(cam.get("snapshot_url") or cam.get("url") or cam.get("rtsp_url")),
                        username=cam.get("username"),
                        password=cam.get("password"),
                        timeout=5
                    )
                    if path and os.path.exists(path):
                        img = cv2.imread(path)
                        if img is not None:
                            sharp = compute_sharpness(img)
                            candidates.append((sharp, path))
                            logger.debug(f"[CAMERA] {oid} snap#{i+1} sharpness={sharp:.2f} -> {path}")
                except Exception as e:
                    logger.error(f"[CAMERA] {oid} snapshot error: {e}")
                time.sleep(0.05)

            ann_path = None
            bbox = None
            if candidates:
                candidates.sort(key=lambda t: t[0], reverse=True)
                best_sharp, best_path = candidates[0]
                label = (f"{obj.get('type','UNKNOWN')} | {obj.get('speed_kmh',0.0):.1f} km/h | "
                         f"{obj.get('distance',0.0):.1f} m | "
                         f"Az:{obj.get('azimuth',0.0):.1f}° El:{obj.get('elevation',0.0):.1f}°")
                try:
                    evt_ts = float(obj.get("timestamp", time.time()))
                except Exception:
                    evt_ts = time.time()
                when = datetime.fromtimestamp(evt_ts)
                # Prefer shared orchestrator timestamp so primary/aux co-locate
                ts = job.get("bundle_ts") or when.strftime("%Y%m%d_%H%M%S_%f")

                prefix = "calib" if job.get("purpose") == "calibration" else "violation"
                try:
                    logger.info(f"[ANNOTATION] {oid} best sharpness={best_sharp:.2f}; annotating {best_path}")
                    conf_thr = float(config.get("annotation_conf_threshold", 0.45))

                    # ---- helpers for "annotate at any cost" retries ----
                    def _preproc(src, out_path):
                        try:
                            img = cv2.imread(src)
                            if img is None:
                                return None
                            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                            y, cr, cb = cv2.split(ycrcb)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            y = clahe.apply(y)
                            ycrcb = cv2.merge([y, cr, cb])
                            img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                            # light gamma normalize
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            mean = max(1.0, float(gray.mean()))
                            gamma = 1.1 if mean < 100 else (0.9 if mean > 160 else 1.0)
                            img = cv2.convertScaleAbs(img, alpha=1.0, beta=0)
                            img = np.clip(((img/255.0) ** (1.0/gamma)) * 255.0, 0, 255).astype("uint8")
                            cv2.imwrite(out_path, img)
                            return out_path if os.path.exists(out_path) else None
                        except Exception:
                            return None

                    def _try_annotate(path, thr, hint):
                        return annotate_speeding_object(
                            image_path=path,
                            radar_distance=max(0.1, min(float(obj.get("distance", 0.0)), 100.0)),
                            label=label,
                            min_confidence=thr,
                            obj_x=float(obj.get("x", 0.0)),
                            obj_y=float(obj.get("y", 0.0)),
                            obj_z=float(obj.get("z", 0.0)),
                            class_hint=str(hint or "").upper()
                        )

                    # attempt sequence
                    ann_path = None; bbox = None; yolo_label = None
                    visual_distance = None; corrected_distance = None
                    attempts = []
                    # 1) best as-is, configured threshold, hinted class
                    attempts.append((best_path, conf_thr, obj.get("type")))
                    # 2) best as-is, relaxed threshold
                    attempts.append((best_path, max(0.30, conf_thr * 0.8), obj.get("type")))
                    # 3) best preprocessed, more relaxed, no hint
                    tmp1 = os.path.join("snapshots", f".anno_{oid}_{ts}_p1.jpg")
                    p1 = _preproc(best_path, tmp1)
                    if p1: attempts.append((p1, max(0.25, conf_thr * 0.7), ""))
                    # 4) second-best as-is (if available)
                    if len(candidates) > 1:
                        second_path = candidates[1][1]
                        attempts.append((second_path, max(0.25, conf_thr * 0.7), obj.get("type")))
                        # 5) second-best preprocessed
                        tmp2 = os.path.join("snapshots", f".anno_{oid}_{ts}_p2.jpg")
                        p2 = _preproc(second_path, tmp2)
                        if p2: attempts.append((p2, 0.22, ""))

                    for (imgp, thr, hint) in attempts:
                        try:
                            a_path, vdist, cdist, bb, lbl = _try_annotate(imgp, thr, hint)
                            if a_path:
                                ann_path, visual_distance, corrected_distance = a_path, vdist, cdist
                                bbox, yolo_label = bb, lbl
                                break
                        except Exception as _e:
                            logger.debug(f"[ANNOTATION] retry failed on {imgp}: {_e}")

                    obj["visual_distance"] = float(visual_distance or 0.0)
                    obj["distance"] = float(corrected_distance or obj.get("distance", 0.0))
                    obj["snapshot_status"] = "valid" if ann_path else "failed"
                    if not ann_path:
                        obj["annotation_error"] = "no_bbox_or_low_confidence"
                    if yolo_label:
                        obj["vehicle_type"] = yolo_label
                        obj["type"] = yolo_label
                except Exception as e:
                    logger.exception(f"[ANNOTATION] {oid} failed: {e}")
                    obj["snapshot_status"] = "failed"
                    obj["annotation_error"] = f"exception:{str(e)[:160]}"

                # --- ANPR (only if we have a valid annotated image) ---
                try:
                    if (ann_path and bbox and str(obj.get("type","")).upper() in {"CAR","BUS","TRUCK","BIKE","BICYCLE","VEHICLE"}):
                        # Run OCR on the original snapshot to avoid plate truncation by the crop.
                        ax1, ay1, ax2, ay2 = bbox
                        anpr = run_anpr(best_path, roi=(ax1, ay1, ax2, ay2), save_dir="snapshots") or {}
                        # populate in-memory fields (for DB/telemetry)
                        if anpr.get("plate_text"):
                            obj["plate_text"] = anpr["plate_text"]
                            obj["plate_conf"] = float(anpr.get("plate_conf", 0.0))
                            obj["plate_bbox"] = anpr.get("plate_bbox")
                            obj["plate_crop_path"] = anpr.get("crop_path")
                            logger.info(f"[ANPR] {oid} -> {obj['plate_text']} ({obj['plate_conf']:.2f})")
                        else:
                            obj["plate_text"] = None
                            obj["plate_conf"] = 0.0
                        # write sidecar next to the annotated image for UI/overlay
                        try:
                            sidecar = os.path.splitext(ann_path)[0] + ".anpr.json"
                            with open(sidecar, "w") as f:
                                json.dump({
                                    "plate_text": obj.get("plate_text") or "",
                                    "plate_conf": float(obj.get("plate_conf", 0.0) or 0.0),
                                    "plate_bbox": obj.get("plate_bbox") or [],
                                    "crop_path":  obj.get("plate_crop_path") or ""
                                }, f, indent=2)
                        except Exception as se:
                            logger.debug(f"[ANPR] sidecar write failed: {se}")
                    else:
                        obj["plate_text"] = None
                        obj["plate_conf"] = 0.0
                except Exception as e:
                    logger.debug(f"[ANPR] skipped due to error: {e}")

                # If we’re in operator calibration mode, record a radar↔image correspondence
                try:
                    if bool(config.get("calibration_mode", False)) and bbox:
                        x1, y1, x2, y2 = bbox
                        u = 0.5 * (float(x1) + float(x2))
                        v = 0.5 * (float(y1) + float(y2))
                        cam = (job.get("camera") or {})
                        # Prefer a real numeric DB camera_id; fall back to name/primary for legacy
                        raw_cam_id = job.get("camera_id") or cam.get("id") or cam.get("name") \
                                     or (config.get("primary_camera_id") or "primary")
                        try:
                            camera_id = str(int(raw_cam_id))
                        except Exception:
                            camera_id = str(raw_cam_id)
                        camera_role = str(cam.get("role") or "primary")
                        add_pair(
                            u=u, v=v,
                            r_m=float(obj.get("distance", obj.get("range", 0.0))),
                            az_deg=float(obj.get("azimuth", 0.0)),
                            el_deg=float(obj.get("elevation", 0.0)),
                            meta={
                                "object_id": obj.get("object_id"),
                                "type": obj.get("type"),
                                "speed_kmh": obj.get("speed_kmh"),
                                "camera_role": camera_role
                            },
                            camera_id=camera_id
                        )
                        try:
                            if best_path:
                                with open(best_path + ".live.json", "w") as _f:
                                    json.dump({"camera_id": camera_id}, _f)
                        except Exception:
                            pass
                except Exception as _e:
                    logger.debug(f"[CALIB] add_pair skipped: {_e}")

                # Only annotated & motion-gated objects get a video clip
                if ann_path and obj.get("snapshot_status") == "valid" and bool(config.get("post_annotation_clip_request", False)):
                    try:
                        evt_ts = float(obj.get("timestamp", time.time()))
                        event_id_local = int(evt_ts * 1000)
                        camera_id = str(
                            cam.get("id")
                            or cam.get("name")
                            or (config.get("primary_camera_id") or "primary")
                        )
                        request_violation_clip_multi(
                            camera_id,
                            event_id=event_id_local,
                            event_time=datetime.fromtimestamp(evt_ts),
                            object_id=str(obj.get("object_id") or "NA"),
                            obj_class=str(obj.get("type") or "UNKNOWN"),
                            speed_kmh=float(obj.get("speed_kmh") or 0.0),
                        )
                        logger.info(f"[CLIPS] queued post-annotation for {oid}")
                    except Exception as ce:
                        logger.debug(f"[CLIPS] request skipped: {ce}")
                # Vision motion gate: reject static boxes (bags/signs)
                try:
                    if ann_path and bbox:
                        # use second-best as 'previous' if available
                        prev_path = candidates[1][1] if len(candidates) > 1 else None
                        if prev_path and os.path.exists(prev_path):
                            prev_img = cv2.imread(prev_path)
                            curr_img = cv2.imread(best_path)
                            thr = float(config.get("vision_motion_threshold", 3.0))
                            score = roi_motion_score(prev_img, curr_img, bbox)
                            logger.debug(f"[MOTION GATE] ROI mean diff = {score:.2f} (thr={thr})")
                            if score < thr:
                                obj["annotation_error"] = f"motion_gate(score={score:.2f}<thr={thr}) — kept"

                except Exception as e:
                    logger.debug(f"[MOTION GATE] skipped: {e}")

                # ---- Finalize bundle ONLY if annotation succeeded ----
                if ann_path:
                    final_path = os.path.join("snapshots", f"{prefix}_{oid}_{ts}.jpg")
                    try:
                        if os.path.lexists(final_path):
                            os.remove(final_path)
                    except Exception:
                        pass
                    try:
                        os.rename(ann_path, final_path)
                        ann_path = final_path
                    except Exception as e:
                        logger.debug(f"[RENAME] {ann_path} -> {final_path} failed: {e}")

                    try:
                        bundle_dir = _bundle_dir(prefix, oid, ts)
                        os.makedirs(bundle_dir, exist_ok=True)
                        # 1) Save RAW (unannotated) only now that we have a valid annotation
                        try:
                            raw_dst = os.path.join(bundle_dir, "raw.jpg")
                            raw_img = cv2.imread(best_path)
                            if raw_img is not None:
                                cv2.imwrite(raw_dst, raw_img)
                        except Exception as re:
                            logger.debug(f"[BUNDLE] raw save skipped: {re}")

                        # 2) Move annotated full frame as image.jpg + symlink back to snapshots/
                        bundle_img = os.path.join(bundle_dir, "image.jpg")
                        try:
                            os.replace(final_path, bundle_img)
                            _ensure_symlink(final_path, bundle_img, rel_start="snapshots")
                            ann_path = final_path  # keep DB/UI pointing at the symlink
                        except Exception as e:
                            logger.debug(f"[BUNDLE] image move failed: {e}")

                        # 3) Plate crop (if any) → plate.jpg + snapshots symlink
                        try:
                            _crop_src = obj.get("plate_crop_path")
                            if _crop_src and os.path.exists(_crop_src):
                                new_crop_snap = os.path.join("snapshots", f"{prefix}_{oid}_{ts}_plate.jpg")
                                try:
                                    os.replace(_crop_src, new_crop_snap)
                                    obj["plate_crop_path"] = new_crop_snap
                                except Exception as re:
                                    logger.debug(f"[BUNDLE] crop rename failed: {re}")
                                    obj["plate_crop_path"] = _crop_src
                                try:
                                    bundle_plate = os.path.join(bundle_dir, "plate.jpg")
                                    os.replace(obj["plate_crop_path"], bundle_plate)
                                    _ensure_symlink(obj["plate_crop_path"], bundle_plate, rel_start="snapshots")
                                except Exception as be:
                                    logger.debug(f"[BUNDLE] plate move/symlink failed: {be}")
                        except Exception as ce:
                            logger.debug(f"[BUNDLE] plate crop handling skipped: {ce}")

                        # 4) Absorb any prepared clip(s) (primary/aux) and then write meta
                        moved = _absorb_all_matching_clips(bundle_dir, oid, _parse_bundle_ts(ts) or datetime.now())
                        clip_base = os.path.basename(moved[0]) if moved else None
                        _write_meta(bundle_dir, obj=obj, image_name="image.jpg",
                                    clip_name=clip_base, camera=cam, frame_meta=meta)
                        try:
                            mpath = os.path.join(bundle_dir, "meta.json")
                            if os.path.exists(mpath):
                                with open(mpath, "r") as f:
                                    _m = json.load(f)
                                if os.path.exists(os.path.join(bundle_dir, "raw.jpg")):
                                    _m["raw"] = "raw.jpg"
                                with open(mpath, "w") as f:
                                    json.dump(_m, f, indent=2)
                        except Exception as me:
                            logger.debug(f"[BUNDLE] meta enrich skipped: {me}")
                        obj["bundle_dir"] = bundle_dir
                        # ── Evidence Seal: initial seal + QR ───────────────────────────────
                        try:
                            from evidence_seal import make_seal, overlay_qr
                            site_name = str(config.get("site", {}).get("name") or "Site")
                            seal = make_seal(bundle_dir, site=site_name, extra={"oid": oid, "ts": ts})
                            seal_id = seal["payload"]["seal_id"]
                            # Stamp a tiny QR that opens /verify/<seal-id> on the saved image
                            overlay_qr(os.path.join(bundle_dir, "image.jpg"), f"/verify/{seal_id}")
                            # Expose to DB layer (columns optional)
                            obj["seal_id"] = seal_id
                            obj["image_sha256"] = (seal["payload"]["files"] or {}).get("image.jpg")
                            obj["clip_sha256"] = (seal["payload"]["files"] or {}).get("clip.mp4")
                            obj["seal_sig"] = seal.get("sig_b64")
                        except Exception as _e:
                            logger.debug(f"[SEAL] initial seal skipped: {_e}")
                    except Exception as e:
                        logger.debug(f"[BUNDLE] finalize failed: {e}")

                # ---- Cleanup all temporary/raw candidates ----
                try:
                    for _, p in candidates:
                        if os.path.exists(p):
                            try: os.remove(p)
                            except Exception: pass
                    # remove preproc temps if created
                    for tmpx in (locals().get("tmp1"), locals().get("tmp2")):
                        if tmpx and os.path.exists(tmpx):
                            try: os.remove(tmpx)
                            except Exception: pass
                except Exception as e:
                    logger.debug(f"[CLEANUP] skipped: {e}")
            else:
                logger.warning(f"[CAMERA] {oid} no valid snapshots captured")
                obj["snapshot_status"] = "failed"

            # Ensure velX/velY/velZ exist for DB logging
            for k in ("velX", "velY", "velZ"):
                obj.setdefault(k, float(obj.get(k, 0.0)))

            obj["snapshot_path"] = ann_path
            if bool(config.get("clips_enabled", True)):
                obj.setdefault("clip_status", "pending")

            # If this was a calibration job, we only needed the bbox to add a (u,v)<->(r,az,el) pair.
            if job.get("purpose") == "calibration":
                logger.info(f"[CALIB] Pair captured for {oid}; skipping DB insert.")
            else:
                db_job = {
                    "obj": obj,
                    "range_profile": list(map(float, meta.get("range_profile", []))),
                    "noise_profile": list(map(float, meta.get("noise_profile", []))),
                    "heatmap": heatmap  # optional, for fallback image if DB insert fails
                }
                try:
                    _db_q.put_nowait(db_job)
                    with _METRICS_LOCK:
                        _METRICS["enq_db"] += 1
                    logger.info(f"[QUEUE] -> DB job enqueued for {oid}")
                except Exception:
                    with _METRICS_LOCK:
                        _METRICS["drop_db"] += 1
                    logger.error(f"[QUEUE] DB queue full; dropping job for {oid}")

        except Exception as e:
            logger.exception(f"[VIOLATION] Worker error: {e}")
        finally:
            with _METRICS_LOCK:
                _METRICS["done_violation"] += 1
            _violation_q.task_done()

# --- Clip → DB updater (async callback from camera.py) ---
def on_clip_done_callback(res):
    """
    Called by camera.ClipWorker thread when a clip finishes.
    We match the most-recent pending row for the same object_id
    and time window, then update clip_* columns.
    """
    try:

        status = getattr(res, "status", None)
        clip_path = getattr(res, "clip_path", None)
        dur = getattr(res, "duration_s", None)
        fps = getattr(res, "fps", None)
        size_b = getattr(res, "size_bytes", None)
        sha = getattr(res, "sha256", None)

        # 1) Prefer manifest: robust and cheap.
        obj_id = None
        match_dt = None
        try:
            if clip_path:
                mpath = os.path.splitext(clip_path)[0] + ".json"
                if os.path.exists(mpath):
                    with open(mpath, "r") as f:
                        man = json.load(f)
                    meta = man.get("meta", man)
                    obj_id = meta.get("object_id") or man.get("object_id") or None
                    et = meta.get("event_time") or man.get("event_time")
                    if et:
                        match_dt = datetime.fromisoformat(et)
        except Exception:
            pass

        # 2) Fallback to filename parse:
        # violation_YYYYMMDD_HHMMSS_<event>_<object_id...>_<CLASS>_<speed>.mp4
        if (obj_id is None or match_dt is None) and clip_path:
            try:
                base = os.path.splitext(os.path.basename(clip_path))[0]
                parts = base.split("_")
                if len(parts) >= 7 and parts[0] == "violation":
                    ts_str = parts[1] + "_" + parts[2]
                    match_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    oid_token = "_".join(parts[4:-2])  # everything between <event> and <CLASS>
                    obj_id = None if oid_token == "NA" else oid_token
            except Exception:
                pass

        with _db_connect() as conn:
            cur = conn.cursor()

            if status == "ready" and clip_path:
                try:
                    _ensure_web_playable_mp4(clip_path)
                except Exception as _e:
                    logger.debug(f"[CLIPS] faststart ensure skipped: {_e}")
                if obj_id and match_dt:
                    # Update the single most-recent pending row for this object
                    cur.execute("""
                        WITH cand AS (
                          SELECT ctid
                          FROM radar_data
                          WHERE object_id = %s
                            AND (clip_status IS NULL OR clip_status = 'pending')
                            AND datetime BETWEEN %s - INTERVAL '45 seconds'
                                             AND %s + INTERVAL '45 seconds'
                          ORDER BY datetime DESC
                          LIMIT 1
                        )
                        UPDATE radar_data r
                        SET clip_path = %s,
                            clip_status = 'ready',
                            clip_duration_s = %s,
                            clip_fps = %s,
                            clip_size_bytes = %s,
                            clip_sha256 = %s
                        FROM cand
                        WHERE r.ctid = cand.ctid;
                    """, (obj_id, match_dt, match_dt,
                          clip_path, dur, fps, size_b, sha))

                    try:
                        # Find the matching bundle and move this clip, then sweep any other *strict* matches.
                        bundle = _find_bundle_for(obj_id, match_dt, prefix="violation")
                        if bundle and os.path.isdir(bundle):
                            if clip_path and os.path.isfile(clip_path):
                                _move_clip_into_bundle(clip_path, bundle)
                            _absorb_all_matching_clips(bundle, obj_id, match_dt)  # uses strict rules now
                            # ── Evidence Seal: re-seal now that clip exists (preserve seal_id) ──
                            try:
                                from evidence_seal import make_seal
                                seal_id = None
                                sp = os.path.join(bundle, "seal.json")
                                if os.path.exists(sp):
                                    try:
                                        seal_id = json.load(open(sp, "r")).get("payload", {}).get("seal_id")
                                    except Exception:
                                        seal_id = None
                                site_name = str(config.get("site", {}).get("name") or "Site")
                                make_seal(
                                    bundle,
                                    site=site_name,
                                    seal_id=seal_id,
                                    extra={"oid": obj_id, "ts": (match_dt.isoformat() if match_dt else None)}
                                )
                            except Exception as _se:
                                logger.debug(f"[SEAL] re-seal skipped: {_se}")
                    except Exception as _e:
                        logger.debug(f"[BUNDLE] post-save absorb skipped: {_e}")

                else:
                    # Fallback: update the most recent pending row
                    cur.execute("""
                        WITH cand AS (
                          SELECT ctid
                          FROM radar_data
                          WHERE (clip_status IS NULL OR clip_status = 'pending')
                          ORDER BY datetime DESC
                          LIMIT 1
                        )
                        UPDATE radar_data r
                        SET clip_path = %s,
                            clip_status = 'ready',
                            clip_duration_s = %s,
                            clip_fps = %s,
                            clip_size_bytes = %s,
                            clip_sha256 = %s
                        FROM cand
                        WHERE r.ctid = cand.ctid;
                    """, (clip_path, dur, fps, size_b, sha))
            else:
                # Mark the most recent pending row as failed (conservative)
                cur.execute("""
                    WITH cand AS (
                      SELECT ctid
                      FROM radar_data
                      WHERE (clip_status IS NULL OR clip_status = 'pending')
                      ORDER BY datetime DESC
                      LIMIT 1
                    )
                    UPDATE radar_data r
                    SET clip_status = 'failed'
                    FROM cand
                    WHERE r.ctid = cand.ctid;
                """)

            conn.commit()
    except Exception as e:
        # We keep this totally non-blocking for the clip worker — just log.
        try:
            logger.warning(f"[CLIPS][DB] callback error: {e}")
        except Exception:
            print(f"[CLIPS][DB] callback error: {e}")

def _db_worker():
    """
    Consumes DB jobs and writes to PostgreSQL.
    On failure: logs CSV fallback and tries to save a heatmap snapshot if provided.
    """
    while True:
        job = _db_q.get()
        try:
            obj = job["obj"]
            oid = obj.get("object_id", "UNKNOWN")
            logger.debug(f"[DB] Writing record for {oid}")

            with _db_connect() as conn:
                cur = conn.cursor()
                obj_ts = to_native(obj.get("timestamp") or time.time())
                measured_dt = datetime.fromtimestamp(obj_ts)
                cur.execute("""
                    INSERT INTO radar_data (
                        timestamp, measured_at, datetime, sensor, object_id, type, confidence, speed_kmh,
                        velocity, distance, direction, signal_level, doppler_frequency, snapshot_path,
                        x, y, z, range, azimuth, elevation, motion_state, snapshot_status,
                        velx, vely, velz, snr, noise,
                        accx, accy, accz,
                        range_profile, noise_profile,
                        plate_text, plate_conf, plate_bbox, plate_crop_path,
                        clip_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                              %s, %s, %s, %s, %s, %s, %s, %s,
                              %s, %s, %s, %s, %s,
                              %s, %s, %s,
                              %s, %s,
                              %s, %s, %s, %s,
                              %s::clip_state)
                """, (
                    obj_ts,
                    measured_dt,
                    measured_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    to_native(obj.get('sensor', 'IWR6843ISK')),
                    to_native(obj.get('object_id')),
                    to_native(obj.get('type')),
                    to_native(obj.get('confidence', 1.0)),
                    to_native(obj.get('speed_kmh', 0.0)),
                    to_native(obj.get('velocity', 0.0)),
                    to_native(obj.get('distance', 0.0)),
                    to_native(obj.get("direction", "unknown")),
                    to_native(obj.get("signal_level", 0.0)),
                    to_native(obj.get("doppler_frequency", 0.0)),
                    to_native(obj.get("snapshot_path")),
                    to_native(obj.get("x", 0.0)),
                    to_native(obj.get("y", 0.0)),
                    to_native(obj.get("z", 0.0)),
                    to_native(obj.get("range", 0.0)),
                    to_native(obj.get("azimuth", 0.0)),
                    to_native(obj.get("elevation", 0.0)),
                    to_native(obj.get("motion_state", "unknown")),
                    to_native(obj.get("snapshot_status", "FAILED")),
                    to_native(obj.get("velX", 0.0)),
                    to_native(obj.get("velY", 0.0)),
                    to_native(obj.get("velZ", 0.0)),
                    to_native(obj.get("snr", 0.0)),
                    to_native(obj.get("noise", 0.0)),
                    to_native(obj.get("accX", 0.0)),
                    to_native(obj.get("accY", 0.0)),
                    to_native(obj.get("accZ", 0.0)),
                    job.get("range_profile", []),
                    job.get("noise_profile", []),
                    obj.get("plate_text"),
                    obj.get("plate_conf", 0.0),
                    json.dumps(obj.get("plate_bbox") or []),
                    obj.get("plate_crop_path"),
                    obj.get("clip_status")
                ))
                conn.commit()
                logger.info(f"[DB] OK -> {oid} | {obj.get('type')} | {obj.get('speed_kmh',0.0):.1f} km/h | {obj.get('snapshot_status')}")
        except Exception as e:
            # Fallbacks
            obj = job.get("obj", {})
            oid = obj.get("object_id", "UNKNOWN")
            logger.error(f"[DB] Insert failed for {oid}: {e}")
            try:
                log_violation_to_csv(obj)
                logger.warning(f"[DB] Fallback CSV logged for {oid}")
            except Exception as e2:
                logger.error(f"[DB] CSV fallback failed for {oid}: {e2}")

            # Save a fallback heatmap image if available
            try:
                H = job.get("heatmap", None)
                if H is not None:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fallback_path = os.path.join("snapshots", f"heatmap_{oid}_{timestamp_str}.jpg")
                    arr = None
                    # Try a few plausible row counts
                    flat = np.array(H).ravel()
                    for rows in (32, 64, 128):
                        if flat.size % rows == 0:
                            try:
                                arr = flat.reshape((rows, -1))
                                break
                            except Exception:
                                pass
                    if arr is not None and arr.size > 0:
                        vmin = np.percentile(arr, 1)
                        vmax = np.percentile(arr, 99)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        im = ax.imshow(arr, cmap='hot', interpolation='nearest', origin='lower',
                                       aspect='auto', vmin=vmin, vmax=vmax)
                        ax.set_title("Range/Doppler Heatmap")
                        fig.colorbar(im, ax=ax, label="Intensity")
                        fig.tight_layout()
                        fig.savefig(fallback_path)
                        plt.close(fig)
                        logger.info(f"[HEATMAP] Fallback saved for {oid} -> {fallback_path}")
                    else:
                        logger.warning(f"[HEATMAP] Fallback skipped; reshape failed for {oid}")
            except Exception as e3:
                logger.warning(f"[HEATMAP] Fallback error for {oid}: {e3}")
        finally:
            with _METRICS_LOCK:
                _METRICS["done_db"] += 1
            _db_q.task_done()

def _heatmap_worker(plotter: Live3DPlotter):
    """
    Coalesces heatmap updates — keeps only the newest payload.
    Payload can be any array-like; plotter.update_heatmap handles storage/writes.
    """
    pending = None
    while True:
        try:
            pending = _heatmap_q.get(timeout=0.5)
            # Drain to keep only the latest
            while True:
                pending = _heatmap_q.get_nowait()
        except Empty:
            pass
        except Exception as e:
            logger.warning(f"[HEATMAP] Worker queue error: {e}")

        if pending is not None:
            try:
                plotter.update_heatmap(pending)  # plotter will handle saving/logging
                with _METRICS_LOCK:
                    _METRICS["done_heatmap"] += 1
            except Exception as e:
                logger.warning(f"[HEATMAP] Update failed: {e}")
            finally:
                pending = None

def _metrics_worker():
    """
    Periodically prints queue sizes & throughput metrics.
    """
    while True:
        time.sleep(5)
        try:
            with _METRICS_LOCK:
                m = dict(_METRICS)
            logger.info(
                "[HEALTH] Qsizes: vio=%d db=%d hmap=%d | "
                "enq(vio/db/hmap)=%d/%d/%d | done(vio/db/hmap)=%d/%d/%d | drops(vio/db/hmap)=%d/%d/%d | "
                "frames=%d classified=%d tracked=%d",
                _violation_q.qsize(), _db_q.qsize(), _heatmap_q.qsize(),
                m["enq_violation"], m["enq_db"], m["enq_heatmap"],
                m["done_violation"], m["done_db"], m["done_heatmap"],
                m["drop_violation"], m["drop_db"], m["drop_heatmap"],
                m["frames"], m["classified"], m["tracked"]
            )
        except Exception:
            logger.warning("[HEALTH] Metrics print failed:\n" + traceback.format_exc())

def _ptz_worker():
    """Consumes PTZ jobs; supports 'preset', 'nudge' (pan/tilt[/zoom]), 'zoom', 'abs/rel', 'stop', lock/unlock."""
    ctl = None
    last_cfg_fingerprint = None
    pose_est = {"pan": 0.0, "tilt": 0.0}
    have_est = False

    def _resolve_cfg(job: dict) -> dict:
        # Priority: explicit → camera.nested → global config["ptz"]
        if "ptz_cfg" in job and job["ptz_cfg"]:
            return job["ptz_cfg"]
        cam = job.get("cam") or {}
        if isinstance(cam, dict):
            if cam.get("ptz"):
                return cam["ptz"]
            # some callers provide creds at top-level cam
            if any(k in cam for k in ("host","ip","username","password","port","profile_token")):
                return {
                    "host": cam.get("host") or cam.get("ip"),
                    "username": cam.get("username"),
                    "password": cam.get("password"),
                    "port": cam.get("port") or 80,
                    "profile_token": cam.get("profile_token","")
                }
        return _ptz_cfg()

    def _ensure_controller(cfg: dict):
        nonlocal ctl, last_cfg_fingerprint
        if not PTZController:
            return None
        # Build a small fingerprint to detect cfg changes
        fp = (
            cfg.get("host") or cfg.get("ip"),
            cfg.get("username"),
            "****" if cfg.get("password") else None,
            int(cfg.get("port") or 80),
            str(cfg.get("profile_token") or "")
        )
        if ctl is None or fp != last_cfg_fingerprint:
            try:
                ctl = PTZController(
                    host=cfg.get("host") or cfg.get("ip"),
                    username=cfg.get("username") or "",
                    password=cfg.get("password") or "",
                    port=int(cfg.get("port") or 80),
                    profile_token=str(cfg.get("profile_token") or ""),
                    invert_pan=bool(cfg.get("invert_pan", False)),
                    invert_tilt=bool(cfg.get("invert_tilt", False)),
                    max_pan_speed=float(cfg.get("max_pan_speed", 0.70)),
                    max_tilt_speed=float(cfg.get("max_tilt_speed", 0.70)),
                    max_zoom_speed=float(cfg.get("max_zoom_speed", 0.50)),
                    min_cmd_interval=float(cfg.get("min_cmd_interval", 0.08)),
                    yaw_offset_deg=float(cfg.get("yaw_offset_deg", 0.0)),
                    tilt_offset_deg=float(cfg.get("tilt_offset_deg", 0.0)),
                    deadband_deg=float(cfg.get("deadband_deg", _ptz_deadband_deg)),
                )
                last_cfg_fingerprint = fp
                logger.info("[PTZ] Controller (re)initialized for %s", fp[0])
            except Exception as e:
                ctl = None
                logger.debug(f"[PTZ] init failed: {e}")
        return ctl

    def _update_estimate(dpan_deg: float, dtilt_deg: float):
        nonlocal pose_est, have_est
        pose_est["pan"]  = float(pose_est.get("pan", 0.0))  + float(dpan_deg or 0.0)
        pose_est["tilt"] = float(pose_est.get("tilt", 0.0)) + float(dtilt_deg or 0.0)
        have_est = True

    def _do_nudge(ctl_obj, vx: float, vy: float, sec: float, vz: float = 0.0, owner: str | None = None):
        # Support multiple method names across camera SDKs
        for name in ("continuous_move", "cont", "move_continuous", "move"):
            fn = getattr(ctl_obj, name, None)
            if callable(fn):
                try:
                    return fn(vx=vx, vy=vy, duration=sec, zoom=vz, owner=owner)
                except TypeError:
                    try:
                        return fn(vx=vx, vy=vy, duration=sec, zoom=vz, owner=owner)
                    except TypeError:
                        try:
                            return fn(vx, vy, sec, vz)
                        except Exception:
                            pass
                    except Exception: pass
        # very generic escape hatch:
        fn = getattr(ctl_obj, "send", None)
        if callable(fn):
            try: return fn("cont", {"vx": vx, "vy": vy, "vz": vz, "sec": sec, "owner": owner})
            except Exception: pass

    def _do_zoom(ctl_obj, v: float, sec: float, owner: str | None = None):
        # Prefer controller's burst helper (Illustra path), else fallbacks.
        zb = getattr(ctl_obj, "zoom_burst", None)
        if callable(zb):
            try:
                return zb(vz=v, duration=sec, owner=owner)
            except Exception:
                pass
        # Try common zoom APIs across different PTZ SDKs
        for name in ("zoom_continuous", "zoom", "set_zoom_velocity", "zoomMove"):
            fn = getattr(ctl_obj, name, None)
            if callable(fn):
                try:
                    # prefer keyword args if supported
                    return fn(v=v, sec=sec)
                except TypeError:
                    try: return fn(v, sec)
                    except Exception: pass
        mv = getattr(ctl_obj, "continuous_move", None)
        if callable(mv):
            try:
                return mv(vx=0.0, vy=0.0, duration=sec, zoom=v)
            except Exception:
                pass
        # generic escape hatch:
        fn = getattr(ctl_obj, "send", None)        
        if callable(fn):
            try: 
                return fn("zoom", {"v": v, "sec": sec})
            except Exception: pass

    while True:
        job = _ptz_q.get()
        try:
            if not isinstance(job, dict):
                continue
            cfg = _resolve_cfg(job)
            if not (cfg and cfg.get("host")):
                continue
            ctl_live = _ensure_controller(cfg)
            if not ctl_live:
                continue
            jtype = job.get("type") or ("preset" if job.get("preset") else None)
            owner = job.get("owner")
            if jtype == "nudge":
                vx = max(-1.0, min(1.0, float(job.get("vx", 0.0))))
                vy = max(-1.0, min(1.0, float(job.get("vy", 0.0))))
                sec = max(0.05, min(2.0, float(job.get("sec", 0.3))))
                vz = max(-1.0, min(1.0, float(job.get("vz", 0.0))))
                _do_nudge(ctl_live, vx, vy, sec, vz, owner)
            elif jtype == "track":
                # Use the controller's driftless relative-step tracker
                try:
                    az = float(job.get("az", 0.0))
                    el = float(job.get("el", 0.0))
                    dist = job.get("dist", None)
                    dur = float(cfg.get("nudge_seconds", _ptz_nudge_sec))
                    # auto_track_step applies offsets, deadband, min step, etc.
                    res = getattr(ctl_live, "auto_track_step")(az_deg=az, el_deg=el, dist_m=dist, owner=owner, duration=dur)
                    # If we were called in plain radar az/el mode, we don't trust estimate — skip integration.
                    if isinstance(res, dict) and "dpan" in res and "dtilt" in res and have_est:
                        _update_estimate(res["dpan"], res["dtilt"])
                except Exception as e:
                    logger.debug(f"[PTZ] track step failed: {e}")
            elif jtype == "track_angles":
                try:
                    pan_des  = float(job.get("pan"))
                    tilt_des = float(job.get("tilt"))
                    dist     = job.get("dist", None)
                except Exception:
                    continue
                # If we have no estimate yet, assume we've been homed.
                if not have_est:
                    # Best-effort: assume current = desired (avoids wild first jump)
                    pose_est["pan"] = float(pan_des)
                    pose_est["tilt"] = float(tilt_des)
                    have_est = True
                # Error = desired - estimated
                err_pan  = float(pan_des  - pose_est.get("pan",  0.0))
                err_tilt = float(tilt_des - pose_est.get("tilt", 0.0))
                dur = float(cfg.get("nudge_seconds", _ptz_nudge_sec))
                try:
                    res = getattr(ctl_live, "auto_track_step")(az_deg=err_pan, el_deg=err_tilt,
                                                               dist_m=dist, owner=owner, duration=dur)
                except Exception as e:
                    logger.debug(f"[PTZ] track_angles step failed: {e}")
                    res = None
                # Integrate returned relative motion to keep our estimate in sync
                try:
                    if isinstance(res, dict) and "dpan" in res and "dtilt" in res:
                        _update_estimate(res["dpan"], res["dtilt"])
                except Exception:
                    pass
                # Optional zoom gate: only when nearly centered
                try:
                    if dist is not None:
                        err_h = max(abs(err_pan), abs(err_tilt))
                        if err_h <= float(cfg.get("zoom_enable_error_deg", _ptz_zoom_enable_err_deg)):
                            # simple P zoom around target distance
                            sm = _PTZ_DIST_LP.get("track", float(dist))
                            sm = (float(_PTZ_DIST_ALPHA) * float(dist)) + (1.0 - float(_PTZ_DIST_ALPHA)) * float(sm)
                            _PTZ_DIST_LP["track"] = sm
                            err_m = float(sm - float(cfg.get("zoom_target_m", _ptz_zoom_target_m)))
                            if abs(err_m) > float(cfg.get("zoom_deadband_m", _ptz_zoom_deadband_m)):
                                vz = float(cfg.get("zoom_kp", _ptz_zoom_kp)) * float(err_m)
                                vz = max(-1.0, min(1.0, vz))
                                _do_zoom(ctl_live, v=vz, sec=float(cfg.get("zoom_seconds", _ptz_zoom_seconds)), owner=owner)
                except Exception:
                    pass
            elif jtype == "zoom":
                vz = max(-1.0, min(1.0, float(job.get("vz", 0.0))))
                sec = max(0.05, min(2.0, float(job.get("sec", 0.3))))
                if abs(vz) > 1e-6:
                    _do_zoom(ctl_live, vz, sec, owner)
            elif jtype in ("abs", "absolute"):
                pan  = job.get("pan",  None)
                tilt = job.get("tilt", None)
                zoom = job.get("zoom", None)
                # Try common absolute move APIs
                for name in ("absolute_move", "move_absolute", "absolute"):
                    fn = getattr(ctl_live, name, None)
                    if callable(fn):
                        try:
                            fn(pan=pan, tilt=tilt, zoom=zoom, owner=owner); 
                            if pan is not None and tilt is not None:
                                pose_est["pan"]  = float(pan)
                                pose_est["tilt"] = float(tilt)
                                have_est = True
                            break
                        except TypeError:
                            try: fn(pan, tilt, zoom); break
                            except Exception: pass
                else:
                    fn = getattr(ctl_live, "send", None)
                    if callable(fn):
                        try: fn("abs", {"pan": pan, "tilt": tilt, "zoom": zoom})
                        except Exception: pass
                    if pan is not None and tilt is not None:
                        pose_est["pan"]  = float(pan); pose_est["tilt"] = float(tilt); have_est = True
            elif jtype in ("rel", "relative"):
                dpan  = float(job.get("dpan", 0.0))
                dtilt = float(job.get("dtilt", 0.0))
                dzoom = float(job.get("dzoom", 0.0))
                for name in ("relative_move", "move_relative", "relative"):
                    fn = getattr(ctl_live, name, None)
                    if callable(fn):
                        try:
                            fn(dpan=dpan, dtilt=dtilt, dzoom=dzoom, owner=owner); break
                        except TypeError:
                            try: fn(dpan, dtilt, dzoom); break
                            except Exception: pass
                else:
                    fn = getattr(ctl_live, "send", None)
                    if callable(fn):
                        try: fn("rel", {"dpan": dpan, "dtilt": dtilt, "dzoom": dzoom})
                        except Exception: pass
                _update_estimate(dpan, dtilt)
            elif jtype == "preset":
                preset = str(job.get("preset"))
                try:
                    ctl_live.goto_preset(preset)
                except Exception as e:
                    logger.debug(f"[PTZ] goto_preset failed: {e}")
            elif jtype == "stop":
                try:
                    # Prefer controller hard stop
                    fn = getattr(ctl_live, "stop", None)
                    if callable(fn):
                        try:
                            fn(hard=True)
                        except TypeError:
                            fn()
                    # Belt-and-suspenders: directly zero raw speeds if exposed
                    cli = getattr(ctl_live, "client", None)
                    raw = getattr(cli, "stop_hard", None)
                    if callable(raw):
                        try:
                            raw()
                        except Exception:
                            pass
                except Exception:
                    pass
                _ptz_pause(_ptz_cfg_val("pause_after_stop_s", 0.7))
            elif jtype == "settle":
                """
                Wait until PTZ is stationary by sampling status speeds (if supported),
                else fallback to a bounded sleep. This makes home-on-enable deterministic.
                """
                to_s   = float(job.get("timeout_s", 0.6))
                samp   = max(1, int(job.get("samples", 2)))
                slp_s  = max(0.05, float(job.get("sleep_s", 0.15)))
                deadline = time.time() + max(0.0, to_s)
                ok = False
                # Prefer controller helper if present
                fn = getattr(ctl_live, "is_settled", None)
                if callable(fn):
                    try:
                        ok = bool(fn(samples=samp, sleep_s=slp_s))
                    except Exception:
                        ok = False
                # Fallback: poll raw client speeds (if exposed)
                if not ok:
                    try:
                        sp_ok = 0
                        while time.time() < deadline:
                            st = getattr(ctl_live, "client", None)
                            st = getattr(st, "status_parsed", None)
                            if callable(st):
                                d = st() or {}
                                pansp  = float(d.get("pansp", 0.0) or 0.0)
                                tiltsp = float(d.get("tiltsp", 0.0) or 0.0)
                                if abs(pansp) <= 1.0 and abs(tiltsp) <= 1.0:
                                    sp_ok += 1
                                    if sp_ok >= samp:
                                        ok = True
                                        break
                                else:
                                    sp_ok = 0
                            time.sleep(slp_s)
                    except Exception:
                        ok = False
                # Final fallback: bounded sleep
                if not ok and to_s > 0:
                    time.sleep(to_s)
            elif jtype == "pose_estimate":
                try:
                    pan = float(job.get("pan"))
                    tilt = float(job.get("tilt"))
                    pose_est["pan"]  = pan
                    pose_est["tilt"] = tilt
                    have_est = True
                except Exception:
                    pass
            elif jtype in ("lock", "unlock", "auto_state"):
                owner = str(job.get("owner") or "auto")
                try:
                    if jtype == "lock" or (jtype == "auto_state" and bool(job.get("enabled", False))):
                        getattr(ctl_live, "lock", lambda *_a, **_k: None)(owner)
                    else:
                        getattr(ctl_live, "unlock", lambda *_a, **_k: None)(owner)
                except Exception:
                    logger.debug(f"[PTZ] {jtype} failed", exc_info=True)
        except PTZError as e:
            logger.debug(f"[PTZ] error: {e}")
        except Exception as e:
            logger.debug(f"[PTZ] unexpected: {e}")
        finally:
            try: _ptz_q.task_done()
            except Exception: pass

def _aux_snap_worker():
    """
    Takes burst snapshots from aux cameras and drops them into the *same*
    violation bundle as the primary (if/when it appears).
    """
    while True:
        job = _aux_snap_q.get()
        try:
            cam: dict = job["camera"]
            obj: dict = job["obj"]
            # Prefer a shared, deterministic bundle timestamp if provided
            bundle_ts: str | None = job.get("bundle_ts")
            ts:  float = float(obj.get("timestamp") or time.time())
            oid: str   = obj.get("object_id") or _allocate_display_id(obj.get("type","unknown"), ts)
            when = datetime.fromtimestamp(ts)

            if bundle_ts:
                # Exact same folder as primary by construction
                bundle = _bundle_dir("violation", oid, bundle_ts)
                os.makedirs(bundle, exist_ok=True)
            else:
                # Legacy: wait briefly for a near-time bundle to appear
                bundle = None
                for _ in range(15):  # ~3s
                    bundle = _find_bundle_for(oid, when, prefix="violation")
                    if bundle and os.path.isdir(bundle):
                        break
                    time.sleep(0.2)
                if not bundle:
                    # Deterministic fallback on local time if primary hasn't created one
                    stamp = when.strftime("%Y%m%d_%H%M%S_%f")
                    bundle = _bundle_dir("violation", oid, stamp)
                    os.makedirs(bundle, exist_ok=True)

            burst_n = int(config.get("burst_frames_per_camera", 3))
            window_s = max(0.3, float(config.get("burst_window_ms", 1200)) / 1000.0)
            per_delay = window_s / max(1, burst_n)
            url = cam.get("snapshot_url") or cam.get("url")
            user = cam.get("username"); pwd = cam.get("password")
            for i in range(burst_n):
                path = capture_snapshot(url, output_dir=bundle, username=user, password=pwd)
                if path:
                    base = os.path.basename(path)
                    newn = f"aux_{cam.get('name', 'cam')}_{i+1:02d}.jpg"
                    try:
                        os.replace(path, os.path.join(bundle, newn))
                    except Exception:
                        pass
                time.sleep(per_delay)
        except Exception as e:
            logger.debug(f"[AUX] worker exception: {e}")
        finally:
            try: _aux_snap_q.task_done()
            except Exception: pass

class _NoPlotter:
    def update(self, *_a, **_k): pass
    def update_heatmap(self, *_a, **_k): pass
    def update_tm(self, *_a, **_k): pass

def _start_workers(plotter, enable_heatmaps: bool = False):
    # Violation pipeline workers (parallelize if needed)
    for _ in range(2):
        Thread(target=_violation_worker, daemon=True).start()
    # DB + Heatmap + Metrics
    Thread(target=_db_worker, daemon=True).start()
    if enable_heatmaps:
        Thread(target=_heatmap_worker, args=(plotter,), daemon=True).start()
    Thread(target=_metrics_worker, daemon=True).start()
    if enable_heatmaps:
        logger.info("[ASYNC] Workers started: 2x violation, 1x db, 1x heatmap, 1x metrics")
    else:
        logger.info("[ASYNC] Workers started: 2x violation, 1x db, 1x metrics (heatmaps disabled)")
    Thread(target=_ptz_worker, daemon=True).start()
    Thread(target=_aux_snap_worker, daemon=True).start()
    try:
        _ptz_write_status(False, None)
    except Exception:
        pass

@atexit.register
def _dump_metrics_on_exit():
    with _METRICS_LOCK:
        summary = json.dumps(_METRICS, indent=2)
    logger.info(f"[EXIT] Final metrics:\n{summary}")
    try:
        shutdown_video_pipeline()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("[START] IWR6843 Radar Detection System (async enabled)")

    RadarInterface, radar_kwargs = _radar_interface_factory(config)
    radar = RadarInterface(**radar_kwargs)
    _ensure_id_counter_table()
    fusion_alpha = float(config.get("fusion_alpha", 0.6))
    max_vehicle_accel = float(config.get("max_vehicle_accel", 4.0))
    v_unamb_cfg = config.get("unambiguous_speed_ms")
    try:
        v_unamb_cfg = None if v_unamb_cfg in (None, "", "None") else float(v_unamb_cfg)
    except Exception:
        v_unamb_cfg = None
    v_unamb = getattr(radar, "v_unamb_ms", None) or v_unamb_cfg or 4.62
    try:
        logger.info(f"[RF] Unambiguous speed = {float(v_unamb):.2f} m/s ({float(v_unamb)*3.6:.1f} km/h)")
    except Exception:
        logger.info("[RF] Unambiguous speed not available; using defaults")

    tracker = ObjectTracker(
        speed_limit_kmh=config.get("dynamic_speed_limits", {}).get("default"),
        speed_limits_map=config.get("dynamic_speed_limits", {}),
        unambiguous_mps=float(v_unamb),
        fusion_alpha=fusion_alpha,
        max_vehicle_accel=max_vehicle_accel,
        min_snr_db=float(config.get("min_snr_db", 3.0)),
        min_signal_level=float(config.get("min_signal_level", 0.0)),
        unwrap_k_max=int(config.get("unwrap_k_max", 1)),
        max_range_m=float(config.get("tracker_max_range_m", 150.0)),
        min_range_m=float(config.get("tracker_min_range_m", 0.20)),
        max_radial_mps=float(config.get("tracker_max_radial_mps", 80.0)),
        max_speed_mps=float(config.get("tracker_max_speed_mps", 80.0)),
        low_snr_margin_db=float(config.get("tracker_low_snr_margin_db", 3.0)),
        cos_guard=float(config.get("cos_guard", 0.60)),
        strict_radial_only=bool(config.get("strict_radial_only", True)),
        disable_unwrap_for_classes=set(config.get("disable_unwrap_for_classes", ["HUMAN","PERSON","PEDESTRIAN"])),
    )
    # Keep explicit setter for clarity/decoupling from RF layer.
    try:
        tracker.set_unambiguous_speed(float(v_unamb))
        logger.info(f"[TRACKER] v_unamb set -> {float(v_unamb):.2f} m/s; α={fusion_alpha:.2f}; a_max={max_vehicle_accel:.2f}; unwrap_k_max={tracker.unwrap_k_max}")
    except Exception:
        logger.warning("[TRACKER] Unable to set unambiguous speed; proceeding with defaults.")

    classifier = ObjectClassifier()

    ws_sink = None
    try:
        viz_ws = dict(config.get("viz_ws") or {})
        ws_enabled = bool(viz_ws.get("enabled", False) or os.getenv("VIZ_WS_URL"))
        if ws_enabled and _socketio:
            ws_url = str(os.getenv("VIZ_WS_URL") or viz_ws.get("url") or "http://127.0.0.1:8000")
            ws_event = str(os.getenv("VIZ_WS_EVENT") or viz_ws.get("event") or "viz_frame")
            ws_ns = str(os.getenv("VIZ_WS_NS") or viz_ws.get("namespace") or "/")
            try:
                _sio_client = _socketio.Client(reconnection=True)
                if ws_ns and ws_ns != "/":
                    _sio_client.connect(ws_url, namespaces=[ws_ns])
                else:
                    _sio_client.connect(ws_url)
                def _ws_sink(kind: str, jpg_bytes: bytes, _cli=_sio_client, _ev=ws_event, _ns=ws_ns):
                    try:
                        payload = {
                            "kind": kind,
                            "jpg_b64": base64.b64encode(jpg_bytes).decode("ascii"),
                            "ts": time.time(),
                        }
                        # namespace "/" can be passed as None in python-socketio
                        _cli.emit(_ev, payload, namespace=(None if _ns == "/" else _ns))
                    except Exception:
                        pass
                ws_sink = _ws_sink
                atexit.register(lambda: (_sio_client.disconnect()
                                         if getattr(_sio_client, "connected", False) else None))
                logger.info(f"[WS] plotter sink connected -> {ws_url} ns='{ws_ns}' ev='{ws_event}'")
            except Exception as _e:
                logger.warning(f"[WS] sink disabled (connect failed): {_e}")
                ws_sink = None
        elif ws_enabled and not _socketio:
            logger.warning("[WS] sink requested but python-socketio is not installed; continuing without WS")
    except Exception as _e:
        logger.debug(f"[WS] sink setup skipped: {_e}")

    plotter = Live3DPlotter(frame_sink=ws_sink) if bool(config.get("enable_plotter", True)) else _NoPlotter()

    ENABLE_HEATMAPS = bool(config.get("enable_heatmaps", False))
    _start_workers(plotter, enable_heatmaps=ENABLE_HEATMAPS)

    # ──────────────────────────────────────────────────────────────────────────
    # Initialize ring + clip workers for PRIMARY and (optionally) AUX cameras
    # ──────────────────────────────────────────────────────────────────────────
    try:
        _init_video_pipelines_for_all()
        logger.info("[CLIPS] Pipelines initialized (aux=%s)",
                    "on" if bool(config.get("aux_clips_enabled", False)) else "off")
    except Exception as e:
        logger.warning(f"[CLIPS] init-all skipped: {e}")

    COOLDOWN_SECONDS = float(config.get("cooldown_seconds", 1.0))
    min_violation_snr_db = float(config.get("min_violation_snr_db", 2.0))
    min_violation_signal = float(config.get("min_violation_signal", 0.0))
    stall_count = 0
    RADAR_TIMEOUT_S = float(config.get("radar_get_timeout", 0.8))
    STALL_REINIT_AFTER = int(config.get("radar_stall_retries", 3))
    WATCHDOG_EXIT_AFTER = float(config.get("watchdog_hard_exit_after", 0))
    global _CAM_MODEL, _CAM_MODEL_MTIME

    try:
        while True:
            # Hot-reload config marker
            if os.path.exists("reload_flag.txt"):
                try:
                    if os.path.exists(_CAM_MODEL_PATH):
                        mtime = os.path.getmtime(_CAM_MODEL_PATH)
                        if (_CAM_MODEL is None) or (_CAM_MODEL_MTIME != mtime):
                            _CAM_MODEL = CameraModel.load(_CAM_MODEL_PATH)
                            _CAM_MODEL_MTIME = mtime
                            logger.info("[PROJECTION] Reloaded camera_model.json")
                except Exception as e:
                    logger.warning(f"[PROJECTION] Reload failed: {e}")
                logger.info("[CONFIG] Reload flag detected -> reloading config")
                config.clear()
                config.update(load_config())
                tracker.speed_limits_map = config.get("dynamic_speed_limits", {})
                try:
                    tracker.fusion_alpha = float(config.get("fusion_alpha", tracker.fusion_alpha))
                except Exception:
                    pass
                try:
                    tracker.a_max = float(config.get("max_vehicle_accel", tracker.a_max))
                except Exception:
                    pass
                try:
                    tracker.unwrap_k_max = int(config.get("unwrap_k_max", tracker.unwrap_k_max))
                except Exception:
                    pass      
                try:
                    tracker.strict_radial_only = bool(
                        config.get("strict_radial_only", getattr(tracker, "strict_radial_only", True))
                    )
                except Exception:
                    pass
                try:
                    tracker.disable_unwrap_for_classes = set(
                        x.upper() for x in (config.get("disable_unwrap_for_classes",
                                         list(getattr(tracker, "disable_unwrap_for_classes", set()))) or [])
                    )
                except Exception:
                    pass          
                # Prefer interface's v_unamb if available; fall back to config
                v_unamb_cfg = config.get("unambiguous_speed_ms")
                try:
                    v_unamb_cfg = None if v_unamb_cfg in (None, "", "None") else float(v_unamb_cfg)
                except Exception:
                    v_unamb_cfg = None
                v_unamb = getattr(radar, "v_unamb_ms", None) or v_unamb_cfg or 4.62
                try:
                    tracker.set_unambiguous_speed(float(v_unamb))
                    logger.info(f"[TRACKER] Reload -> v_unamb={float(v_unamb):.2f} m/s; α={tracker.fusion_alpha:.2f}; a_max={tracker.a_max:.2f}; unwrap_k_max={tracker.unwrap_k_max}")
                except Exception:
                    pass
                try:
                    _new = try_load_active_model(_selected_camera_id())
                    if _new is not None:
                        _CAM_MODEL = _new
                        logger.info("[PROJECTION] Active camera model refreshed for selected camera")
                except Exception:
                    logger.debug("[PROJECTION] active-camera reload skipped", exc_info=True)
                os.remove("reload_flag.txt")
            try:
                if os.path.exists(_CAM_MODEL_PATH):
                    mtime = os.path.getmtime(_CAM_MODEL_PATH)
                    if (_CAM_MODEL is None) or (_CAM_MODEL_MTIME != mtime):
                        _CAM_MODEL = CameraModel.load(_CAM_MODEL_PATH)
                        _CAM_MODEL_MTIME = mtime
                        logger.info("[PROJECTION] Reloaded camera_model.json (mtime change)")
            except Exception as e:
                logger.debug(f"[PROJECTION] Mtime check failed: {e}")

            frame = get_targets_with_timeout(radar, timeout_s=RADAR_TIMEOUT_S)
            if frame is None:
                stall_count += 1
                if WATCHDOG_EXIT_AFTER > 0 and (stall_count * RADAR_TIMEOUT_S) >= WATCHDOG_EXIT_AFTER:
                    logger.error(f"[WATCHDOG] stalled for ~{stall_count * RADAR_TIMEOUT_S:.1f}s -> exiting")
                    os._exit(21)
                if stall_count >= STALL_REINIT_AFTER:
                    logger.warning(f"[RADAR] get_targets timed out {stall_count}× -> reinitializing interface")
                    try:
                        radar = RadarInterface(**radar_kwargs)
                    except Exception as e:
                        logger.error(f"[RADAR] reinit failed: {e}")
                    stall_count = 0
                time.sleep(0.02)
                continue
            else:
                stall_count = 0
            _ptz_apply_state()
            with _METRICS_LOCK:
                _METRICS["frames"] += 1

            # Heatmap (enqueue only) — accept any FW field that looks like a heatmap
            heatmap = None
            found_key = None
            # 1) known keys
            if bool(ENABLE_HEATMAPS):
                for key in ("range_doppler_heatmap", "range_azimuth_heatmap", "azimuth_elevation_heatmap",
                            "rangeDopplerHeatmap", "rangeAzimuthHeatmap", "azElHeatmap",
                            "rd_heatmap", "ra_heatmap", "ae_heatmap"):
                    h = frame.get(key)
                    if h is None:
                        continue
                    try:
                        if (isinstance(h, np.ndarray) and h.size > 0) or \
                        (isinstance(h, (list, tuple)) and len(h) > 0) or \
                        (isinstance(h, (bytes, bytearray, memoryview)) and len(h) > 0):
                            heatmap = h
                            found_key = key
                            break
                    except Exception:
                        continue
                # 2) fallback: any key that contains "heatmap"
                if heatmap is None:
                    for k, v in list(frame.items()):
                        try:
                            if "heatmap" in str(k).lower():
                                if (isinstance(v, np.ndarray) and v.size > 0) or \
                                   (isinstance(v, (list, tuple)) and len(v) > 0) or \
                                   (isinstance(v, (bytes, bytearray, memoryview)) and len(v) > 0):
                                    heatmap = v
                                    found_key = k
                                    break
                        except Exception:
                            continue
                if heatmap is not None and found_key:
                    try:
                        sz = (heatmap.size if isinstance(heatmap, np.ndarray)
                              else (len(heatmap) if hasattr(heatmap, "__len__") else -1))
                        logger.debug(f"[HEATMAP] detected key='{found_key}' payload_size={sz}")
                    except Exception:
                        logger.debug(f"[HEATMAP] detected key='{found_key}'")

            if ENABLE_HEATMAPS and heatmap is not None:
                try:
                    _heatmap_q.put_nowait(heatmap)
                    with _METRICS_LOCK:
                        _METRICS["enq_heatmap"] += 1
                except Exception:
                    with _METRICS_LOCK:
                        _METRICS["drop_heatmap"] += 1
                    logger.debug("[HEATMAP] queue full; dropping heatmap")

            # Targets
            targets = frame.get("trackData", []) or []
            if not targets:
                pts = frame.get("pointCloud")
                if pts is None:
                    pts = frame.get("point_cloud")
                pts_nonempty = (
                    (isinstance(pts, np.ndarray) and pts.size > 0) or
                    (isinstance(pts, (list, tuple)) and len(pts) > 0)
                )
                if pts_nonempty:
                    try:
                        targets = derive_targets_from_pointcloud(pts)
                        # ---- quality gate for PC clusters (unified pipeline, no DB split) ----
                        _pc_min_n    = int(config.get("pc_min_cluster", 6))
                        _pc_min_snr  = float(config.get("pc_min_snr_db", 6.0))
                        _pc_max_dstd = float(config.get("pc_max_doppler_std_hz", 60.0))
                        _pc_req_dop  = bool(config.get("pc_require_doppler", False))
                        filtered = []
                        for t in (targets or []):
                            n   = int(t.get("pc_n", 1))
                            snr = float(t.get("snr", 0.0))
                            df  = float(t.get("doppler_frequency", 0.0))
                            dsd = float(t.get("pc_doppler_std", 0.0))
                            if n < _pc_min_n:           continue
                            if snr < _pc_min_snr:       continue
                            if _pc_req_dop and abs(df) <= 0.0:  continue
                            if abs(df) > 0.0 and dsd > _pc_max_dstd:  continue
                            filtered.append(t)
                        targets = filtered
                        logger.debug(f"[POINTCLOUD] Kept {len(targets)} high-quality clusters")
                    except Exception as e:
                        logger.warning(f"[POINTCLOUD] Derivation failed: {e}")
                        time.sleep(0.05)
                if (not targets) and (heatmap is not None):
                    hm_targets = derive_target_from_heatmap(heatmap)
                    if hm_targets:
                        targets = hm_targets
                        logger.debug("[HEATMAP] Derived 1 fallback target from RA heatmap")                    
                else:
                    time.sleep(0.05)
                    continue

            now = time.time()
            # Normalize and enrich
            for obj in targets:
                obj["x"] = float(obj.get("x", obj.get("posX", 0.0)))
                obj["y"] = float(obj.get("y", obj.get("posY", 0.0)))
                obj["z"] = float(obj.get("z", obj.get("posZ", 0.0)))
                obj["range"] = float(obj.get("distance", 0.0))

                # Angles
                if "azimuth" not in obj or "elevation" not in obj:
                    x, y, z = obj["x"], obj["y"], obj["z"]
                    obj["azimuth"]   = round(np.degrees(np.arctan2(x, y)), 2)
                    obj["elevation"] = round(np.degrees(np.arctan2(z, np.hypot(x, y))), 2)

                # RF fields
                if "doppler_frequency" not in obj and "doppler" in obj:
                    obj["doppler_frequency"] = float(obj.get("doppler", 0.0))
                obj["doppler_frequency"] = float(obj.get("doppler_frequency", 0.0))

                if "signal_level" not in obj and "signal" in obj:
                    obj["signal_level"] = float(obj.get("signal", 0.0))
                obj["signal_level"] = float(obj.get("signal_level", 0.0))

                # Velocity vector
                vx = float(obj.get("velX", obj.get("vx", 0.0)))
                vy = float(obj.get("velY", obj.get("vy", 0.0)))
                vz = float(obj.get("velZ", obj.get("vz", 0.0)))
                obj["velocity_vector"] = [vx, vy, vz]

                # Do not pass object_id from radar. Let the tracker assign persistent TRK IDs.
                obj.pop("object_id", None)

                # Source/TID labeling
                obj["source"] = obj.get("source", "")
                if obj["source"] != "pointcloud" and "id" in obj:
                    try:
                        obj["source_id"] = f"TID_{int(obj['id'])}"
                    except Exception:
                        obj["source_id"] = f"TID_{str(obj['id'])}"
                elif obj["source"] == "pointcloud" and "pc_id" in obj:
                    obj["source_id"] = f"PC_{int(obj['pc_id'])}"

                # Common fields
                obj.update({
                    "timestamp": now,
                    "sensor": "IWR6843ISK",
                    "distance": float(obj.get("distance", obj.get("range", 0.0))),
                    "velocity": float(np.linalg.norm([vx, vy, vz])) if float(obj.get("velocity", 0.0)) <= 0.0 else float(obj.get("velocity")),
                    "speed_kmh": float(np.linalg.norm([vx, vy, vz]) * 3.6) if float(obj.get("speed_kmh", 0.0)) <= 0.0 else float(obj.get("speed_kmh")),
                    "direction": obj.get("direction", "unknown"),
                    "motion_state": obj.get("motion_state", "unknown"),
                    "confidence": float(obj.get("confidence", 1.0)),
                    "snapshot_status": "PENDING",
                })

                try:
                    motion  = str(obj.get("motion_state", "")).upper()
                    dir_lbl = str(obj.get("direction", "")).upper()
                    df_hz   = float(obj.get("doppler_frequency", 0.0))
                    vx = float(obj.get("velX", 0.0)); vy = float(obj.get("velY", 0.0)); vz = float(obj.get("velZ", 0.0))
                    vec_mps = float(np.linalg.norm([vx, vy, vz]))
                    # Optional conservative static clamp before tracking
                    if bool(config.get("pretrack_static_clamp", False)) and (
                        (motion in ("STATIC", "STATIONARY")) or (dir_lbl == "STATIC")
                    ):
                        eps = float(config.get("static_speed_epsilon_mps", 0.15))
                        if (vec_mps < eps) and (abs(df_hz) < _DOP_MIN_HZ):
                            obj["velocity"]  = 0.0
                            obj["speed_kmh"] = 0.0
                    # Human sanity cap — better to miss than mark impossible speeds
                    typ = str(obj.get("type", "")).upper()
                    human_cap = float(config.get("human_speed_cap_kmh", 35.0))
                    if typ == "HUMAN" and float(obj.get("speed_kmh", 0.0)) > human_cap:
                        obj["velocity"] = 0.0
                        obj["speed_kmh"] = 0.0
                except Exception:
                    pass
                _sanitize_obj_physics(obj)
            # Classify
            # --- Fill in SNR/Noise from point cloud when track SNR is missing/zero ---
            try:
                def _attach_quality_from_points(frame, objs, radius=0.8):
                    pts = frame.get("pointCloud")
                    if pts is None:
                        pts = frame.get("point_cloud")
                    if pts is None:
                        return
                    A = np.asarray(pts, dtype=float)
                    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] < 6:
                        return
                    X = A[:, :3]; S = A[:, 4]; N = A[:, 5]
                    for o in objs:
                        pos = np.array([o.get("x", 0.0), o.get("y", 0.0), o.get("z", 0.0)], dtype=float)
                        d = np.linalg.norm(X - pos, axis=1)
                        m = d <= radius
                        if np.any(m):
                            snr_med = float(np.median(S[m]))
                            noise_med = float(np.median(N[m]))
                            if float(o.get("snr", 0.0)) <= 0.0:
                                o["snr"] = snr_med
                            if float(o.get("noise", 0.0)) <= 0.0:
                                o["noise"] = noise_med
                            if float(o.get("signal_level", 0.0)) <= 0.0:
                                o["signal_level"] = float(o.get("gain", 1.0)) * float(o.get("snr", snr_med))
                _attach_quality_from_points(frame, targets)
            except Exception as _e:
                logger.debug(f"[QUALITY] attach from points skipped: {_e}")
            classifier = classifier or ObjectClassifier()
            classified = classifier.classify_objects(targets)
            with _METRICS_LOCK:
                _METRICS["classified"] += len(classified)
            logger.info(f"[CLASSIFY] {len(classified)} objects")

            # Human-readable prints (pre-track)
            for obj in classified:
                vv = obj.get('velocity_vector', [obj.get('velX', 0.0), obj.get('velY', 0.0), obj.get('velZ', 0.0)])
                acc = obj.get('acceleration', [obj.get('accX', 0.0), obj.get('accY', 0.0), obj.get('accZ', 0.0)])
                df = obj.get('doppler_frequency', obj.get('doppler', 0.0))
                sl = obj.get('signal_level', obj.get('signal', 0.0))
                print("\n---------- Radar Object (pre-track) ----------")
                if 'source_id' in obj: print(f"Source ID: {obj.get('source_id')}")
                elif 'id' in obj:      print(f"Source ID: {obj.get('id')}")
                print(f"Type: {obj.get('type','N/A')} | Conf: {obj.get('confidence',0.0):.2f}")
                print(f"Speed: {obj.get('speed_kmh',0.0):.2f} km/h | Vel: {obj.get('velocity',0.0):.2f} m/s")
                print(f"Dist: {obj.get('distance',0.0):.2f} m | Pos: x={obj.get('x',0.0):.2f} y={obj.get('y',0.0):.2f} z={obj.get('z',0.0):.2f}")
                print(f"VelVec: vx={vv[0]:.2f} vy={vv[1]:.2f} vz={vv[2]:.2f} | Acc: ax={acc[0]:.2f} ay={acc[1]:.2f} az={acc[2]:.2f}")
                print(f"Az:{obj.get('azimuth',0.0):.2f}° El:{obj.get('elevation',0.0):.2f}° | SNR:{obj.get('snr',0.0):.1f}dB")
                print(f"Doppler:{df:.2f}Hz | Signal:{sl:.1f} | Motion:{obj.get('motion_state','UNKNOWN')} Dir:{obj.get('direction','UNKNOWN')}")

            # Track
            V_MAX = float(config.get("pretrack_safe_max_vel_ms", 20.0))   # ~72 km/h hard cap before tracking
            R_MAX = float(config.get("pretrack_safe_max_range_m", 120.0)) # keep close-field only for people
            R_MIN = float(config.get("tracker_min_range_m", 0.20))
            norm = []
            for det in classified:
                d = dict(det)
                d['x'] = float(det.get('x',  det.get('posX', 0.0)))
                d['y'] = float(det.get('y',  det.get('posY', 0.0)))
                d['z'] = float(det.get('z',  det.get('posZ', 0.0)))
                vv = det.get('velocity_vector', [det.get('velX', 0.0), det.get('velY', 0.0), det.get('velZ', 0.0)])
                d['initial_velocity'] = [float(v) for v in vv]
                rng = (d['x']**2 + d['y']**2 + d['z']**2) ** 0.5
                vmag = (vv[0]**2 + vv[1]**2 + vv[2]**2) ** 0.5
                if (not math.isfinite(rng)) or (not math.isfinite(vmag)) or rng < R_MIN or rng > R_MAX or vmag > V_MAX:
                    continue
                if 'doppler_frequency' not in d and 'doppler' in d:
                    d['doppler_frequency'] = float(det.get('doppler', 0.0))
                if 'signal_level' not in d and 'signal' in d:
                    d['signal_level'] = float(det.get('signal', 0.0))
                # Carry source/source_id; tracker will assign persistent object_id
                d['source'] = det.get('source', d.get('source', ''))
                if d['source'] != 'pointcloud' and 'id' in det:
                    try:
                        d['source_id'] = f"TID_{int(det['id'])}"
                    except Exception:
                        d['source_id'] = f"TID_{str(det['id'])}"
                elif d['source'] == 'pointcloud' and 'pc_id' in det:
                    d['source_id'] = f"PC_{int(det['pc_id'])}"
                d.pop('object_id', None)
                norm.append(d)

            tracked = tracker.update_tracks(norm, frame_timestamp=now)

            # Mark currently active track IDs and update last-seen timestamps
            active_tids = set()
            for o in tracked:
                # unify speed/velocity fields (prefers along-road when safe)
                try:
                    _apply_canonical_velocity(o)
                except Exception:
                    pass
                # post-class sanity cap using dynamic_speed_limits (same for all sources)
                try:
                    obj_type = str(o.get("type","")).upper()
                    cap_map  = config.get("dynamic_speed_limits", {})
                    base_cap = cap_map.get(obj_type, cap_map.get("default"))
                    if base_cap is not None:
                        cap_factor = float(config.get("sanity_cap_factor", 1.75))
                        hard_cap   = max(5.0, float(base_cap) * cap_factor)
                        if float(o.get("speed_kmh", 0.0)) > hard_cap:
                            o["speed_kmh"] = float(hard_cap)
                            o["velocity"]  = float(hard_cap) / 3.6
                except Exception:
                    pass
                # Ensure we have a stable tracker ID separate from display ID
                trk_id = o.get('track_id') or o.get('object_id') or f"TRK_{o.get('id', 'UNKNOWN')}"
                o['track_id'] = trk_id
                o.setdefault('tid', trk_id)
                active_tids.add(trk_id)
                _LAST_SEEN_TRK[trk_id] = now

                # Prefer tracked velocity vector
                vx = float(o.get("velX", o.get("vx", 0.0)))
                vy = float(o.get("velY", o.get("vy", 0.0)))
                vz = float(o.get("velZ", o.get("vz", 0.0)))
                if float(o.get("speed_kmh", 0.0)) <= 0.0:
                    spd = (vx*vx + vy*vy + vz*vz) ** 0.5 * 3.6
                    o["velocity"]  = float((vx*vx + vy*vy + vz*vz) ** 0.5)
                    o["speed_kmh"] = float(spd)
                if "speed_along_road_kmh" not in o:
                    ra = getattr(tracker, "road_axis_unit", [0.0, 1.0, 0.0])
                    v_along = vx*ra[0] + vy*ra[1] + vz*ra[2]
                    o["speed_along_road_kmh"] = float(abs(v_along) * 3.6)

                # Motion label from tracked speed (override stale pre-track label)
                spd = float(o.get("speed_along_road_kmh", o.get("speed_kmh", 0.0)) or 0.0)
                mv_thr  = float(config.get("moving_threshold_kmh", 0.8))
                st_thr  = float(config.get("stationary_threshold_kmh", 0.3))
                if spd > mv_thr:
                    o["motion_state"] = "MOVING"
                elif spd < st_thr:
                    o["motion_state"] = "STATIONARY"
                else:
                    o["motion_state"] = o.get("motion_state", "unknown") or "unknown"

                # Direction fallback from tracked velocity if still unknown
                if not o.get("direction") or o.get("direction") == "unknown":
                    ra = getattr(tracker, "road_axis_unit", [0.0, 1.0, 0.0])
                    v_along = vx*ra[0] + vy*ra[1] + vz*ra[2]
                    o["direction"] = "TOWARDS" if v_along < -0.05 else ("AWAY" if v_along > 0.05 else "STATIC")

            # Prune mappings/buffers for tracks that have gone stale (ID reuse protection)
            stale = [tid for tid, ts in list(_LAST_SEEN_TRK.items()) if (now - ts) > TRACK_TTL_SECONDS]
            for tid in stale:
                _LAST_SEEN_TRK.pop(tid, None)
                _TRK_TO_DISPLAY.pop(tid, None)
                speeding_buffer.pop(tid, None)
                acceleration_cache.pop(tid, None)
                violation_state.pop(tid, None)
                last_trigger_time.pop(tid, None)

            with _METRICS_LOCK:
                _METRICS["tracked"] += len(tracked)
            logger.info(f"[TRACK] {len(tracked)} active tracks")
            _ptz_auto_follow(tracked)

            try:
                # 1) point cloud → Nx3 XYZ
                pc = _first_nonempty(frame, ("pointCloud", "point_cloud", "points"), default=[])
                try:
                    XYZ, _D, _S, _N = _extract_xyz_doppler(pc)
                except Exception:
                    XYZ = None
                # 2) tracked objects + points
                plotter.update(tracked, XYZ)
                # 3) TM lane overlays, if present from parser
                tm_viz = frame.get("tm_viz")
                if isinstance(tm_viz, dict):
                    plotter.update_tm(tm_viz)
            except Exception as e:
                logger.debug(f"[PLOTTER] update skipped: {e}")

            # ── Live candidates export (robust path + 1s stickiness) ─────────────
            def _live_dir():
                """Cross-platform live-export directory that's writable."""
                # 1) explicit override always wins
                override = os.environ.get("IWR6843ISK_LIVE_DIR", "").strip()
                if override:
                    p = Path(override).expanduser()
                    try:
                        p.mkdir(parents=True, exist_ok=True)
                        (p / ".touch").write_text("x")
                        (p / ".touch").unlink(missing_ok=True)
                        return p
                    except Exception:
                        pass
                if os.name == "nt":
                    # Use LOCALAPPDATA (or TEMP) on Windows
                    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "C:\\Temp"
                    p = Path(base) / "iwr6843isk"
                    try:
                        p.mkdir(parents=True, exist_ok=True)
                        (p / ".touch").write_text("x")
                        (p / ".touch").unlink(missing_ok=True)
                        return p
                    except Exception:
                        return Path(os.getcwd()) / "runtime" / "iwr6843isk"
                # POSIX fallbacks
                uid = None
                try:
                    uid = os.getuid()  # may not exist on Windows
                except Exception:
                    uid = None
                candidates = [
                    Path(f"/run/user/{uid}/iwr6843isk") if uid is not None else None,
                    Path("/run/iwr6843isk"),
                    Path("/tmp/iwr6843isk"),
                ]
                for d in candidates:
                    if not d:
                        continue
                    try:
                        d.mkdir(parents=True, exist_ok=True)
                        (d / ".touch").write_text("x")
                        (d / ".touch").unlink(missing_ok=True)
                        return d
                    except Exception:
                        continue
                # Last resort: a local folder
                p = Path(os.getcwd()) / "runtime" / "iwr6843isk"
                p.mkdir(parents=True, exist_ok=True)
                return p

            # Keep the last non-empty export for a short time to avoid transient empties
            if not hasattr(sys.modules[__name__], "_LAST_NONEMPTY_EXPORT"):
                _LAST_NONEMPTY_EXPORT = {"t": 0.0, "objects": []}

            try:
                live_dir = _live_dir()
                _LIVE_PATH = live_dir / "live_objects.json"
                tnow = time.time()
                export = []
                for o in tracked:
                    x = float(o.get("x", o.get("posX", 0.0)))
                    y = float(o.get("y", o.get("posY", 0.0)))
                    z = float(o.get("z", o.get("posZ", 0.0)))
                    dist = float(o.get("distance", (x*x + y*y + z*z) ** 0.5))
                    az   = float(o.get("azimuth",  math.degrees(math.atan2(x, max(1e-6, y)))))
                    el   = float(o.get("elevation", math.degrees(math.atan2(z, max(1e-6, (x*x + y*y) ** 0.5)))))
                    export.append({
                        "track_id": o.get("track_id") or o.get("object_id"),
                        "distance_m": dist,
                        "azimuth_deg": az,
                        "elevation_deg": el,
                        "speed_kmh": float(_safe_speed_kmh(o.get("speed_kmh", 0.0)) or 0.0),
                        "snr": float(o.get("signal_level", o.get("snr", 0.0))),
                    })
                # Update sticky cache if we have something
                if export:
                    _LAST_NONEMPTY_EXPORT = {"t": tnow, "objects": export}
                    payload = {"t": tnow, "objects": export}
                else:
                    # Use last non-empty within 1.0s window
                    if (tnow - _LAST_NONEMPTY_EXPORT.get("t", 0.0)) <= 1.0 and _LAST_NONEMPTY_EXPORT["objects"]:
                        payload = {"t": tnow, "objects": _LAST_NONEMPTY_EXPORT["objects"]}
                    else:
                        payload = {"t": tnow, "objects": []}

                tmp = _LIVE_PATH.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, _LIVE_PATH)
            except Exception:
                logger.debug("[CALIB-LIVE] export skipped", exc_info=True)

            # Bind display IDs (once per track), keep TRK for internal gating
            for obj in tracked:
                # Robust: ensure a tracker ID exists even if a prior step didn't set it
                trk_id = str(obj.get('track_id') or obj.get('object_id') or f"TRK_{obj.get('id','UNKNOWN')}")
                obj['track_id'] = trk_id
                obj.setdefault('tid', trk_id)
                disp = _TRK_TO_DISPLAY.get(trk_id)
                if not disp:
                    tstamp = float(obj.get('timestamp', now))
                    disp = _allocate_display_id(obj.get('type', 'unknown'), tstamp)
                    _TRK_TO_DISPLAY[trk_id] = disp
                    logger.info(f"[DISPLAY-ID] NEW {trk_id} -> {disp}")
                # Use display ID for UI/DB/filenames:
                obj['object_id'] = disp

            # Violation logic → enqueue only (internals keyed on TRK, not display ID)
            for obj in tracked:
                oid = obj['object_id']          # display id (for logs / UI / DB)
                tid = obj.get('track_id', oid)  # TRK_xxxxx (for buffers/cooldowns)
                now = time.time()
                vv = obj.get('velocity_vector') or [obj.get('velX', 0.0), obj.get('velY', 0.0), obj.get('velZ', 0.0)]
                if 'velocity' not in obj or obj.get('velocity', 0.0) == 0.0:
                    obj['velocity'] = float(np.linalg.norm(vv))
                if 'speed_kmh' not in obj or obj.get('speed_kmh', 0.0) == 0.0:
                    obj['speed_kmh'] = obj['velocity'] * 3.6
                _sanitize_obj_physics(obj)
                # Acceleration cache
                acc_mag = np.linalg.norm([obj.get("accX", 0.0), obj.get("accY", 0.0), obj.get("accZ", 0.0)])
                acceleration_cache[tid].append(acc_mag)

                speed_limit = tracker.get_limit_for(obj.get("type", "UNKNOWN"))
                speeding = obj["speed_kmh"] > speed_limit
                recent = speeding_buffer[tid]

                if speeding:
                    recent.append(now)
                else:
                    recent.clear()

                acc_buffer = acceleration_cache[tid]
                acc_threshold = config.get("acceleration_threshold", 2.0)
                acc_required_frames = config.get("min_acc_violation_frames", 3)
                acceleration_violating = (
                    len(acc_buffer) >= acc_required_frames and
                    sum(1 for a in acc_buffer if a > acc_threshold) >= acc_required_frames
                )

                limit = max(1.0, float(tracker.get_limit_for(obj.get("type", "default"))))
                margin = float(config.get("violation_margin_kmh", 1.0))
                enter_thresh = limit + margin
                exit_thresh  = max(0.0, limit - margin)

                # Build/maintain per-obj history
                buf = speeding_buffer[tid]
                if obj["speed_kmh"] > enter_thresh:
                    buf.append(now)
                else:
                    # decay / keep only last few if under threshold
                    while buf and (now - buf[0]) > 1.5:
                        buf.popleft()

                min_frames = int(config.get("min_speed_violation_frames", 3))
                confirmed = (len(buf) >= min_frames and (now - buf[0]) <= 1.5)

                # Finite-state machine
                state = violation_state.get(tid, {"active": False, "last_change": 0.0})

                # Enter condition (edge)
                enter = (not state["active"]) and confirmed
                quality_ok = (float(obj.get("snr", 0.0)) >= min_violation_snr_db) or \
                             (float(obj.get("signal_level", 0.0)) > min_violation_signal)
                enter = (not state["active"]) and confirmed and quality_ok
                # Exit condition (hysteresis)
                exit_ = state["active"] and (obj["speed_kmh"] <= exit_thresh)

                if enter:
                    state["active"] = True
                    state["last_change"] = now
                    violation_state[tid] = state

                    # Cooldown check
                    if (now - last_trigger_time.get(tid, 0.0)) >= COOLDOWN_SECONDS:
                        last_trigger_time[tid] = now
                        logger.info(f"[QUEUE] Enqueue violation (edge) for {tid} — {obj['type']} @ {obj['speed_kmh']:.1f} km/h")
                        try:
                            cams = _active_cameras()
                            primary = _select_primary(cams)
                            if primary:
                                _ptz_q.put_nowait({"type": "stop", "cam": dict(primary)})
                                _ptz_pause(_ptz_cfg_val("pause_after_violation_s", 1.0))
                        except Exception:
                            logger.debug("[PTZ] enqueue failed", exc_info=True)

                        try:
                            # Lock all camera media to the exact **enter** instant for this violation
                            _o2 = dict(obj)
                            _o2["timestamp"] = float(state.get("last_change") or now)
                            get_orchestrator().dispatch(
                                _o2,
                                heatmap=heatmap,
                                frame_meta={
                                    "range_profile": frame.get("range_profile", []),
                                    "noise_profile": frame.get("noise_profile", []),
                                },
                            )
                        except Exception:
                            logger.exception("[QUEUE] Orchestrator dispatch failed")

                elif exit_:
                    state["active"] = False
                    state["last_change"] = now
                    violation_state[tid] = state
                # --- Calibration: take snapshots even if not speeding ---
                if bool(config.get("calibration_mode", False)):
                    calib_interval = float(config.get("calibration_snap_interval", 1.0))
                    min_kmh = float(config.get("calibration_min_speed_kmh", 0.1))

                    if obj.get("speed_kmh", 0.0) >= min_kmh and (now - last_calib_shot[tid]) >= calib_interval:
                        last_calib_shot[tid] = now

                        # Select active camera (same logic as violation path)
                        cams = config.get("cameras")
                        if isinstance(cams, list) and cams:
                            idx = int(config.get("selected_camera", 0))
                            cam_cfg = cams[min(max(idx, 0), len(cams) - 1)]
                        elif isinstance(cams, dict):
                            cam_cfg = cams
                        else:
                            cam_cfg = {}

                        camera_payload = {
                            "id": cam_cfg.get("id"),
                            "name": cam_cfg.get("name"),
                            "role": cam_cfg.get("role") or "calibration",
                            "url": cam_cfg.get("snapshot_url") or cam_cfg.get("url"),
                            "username": cam_cfg.get("username"),
                            "password": cam_cfg.get("password"),
                        }
                        cam_id_for_pairs = str(
                            cam_cfg.get("id") or cam_cfg.get("name") or (config.get("primary_camera_id") or "primary")
                        )

                        frame_meta = {
                            "range_profile": frame.get("range_profile", []),
                            "noise_profile": frame.get("noise_profile", []),
                        }

                        try:
                            _violation_q.put_nowait({
                                "obj": dict(obj),
                                "camera": camera_payload,
                                "frame_meta": frame_meta,
                                "heatmap": heatmap,
                                "purpose": "calibration",
                                "camera_id": cam_id_for_pairs
                            })
                            with _METRICS_LOCK:
                                _METRICS["enq_violation"] += 1
                        except Exception:
                            with _METRICS_LOCK:
                                _METRICS["drop_violation"] += 1
                            logger.warning(f"[QUEUE] Calibration queue full; dropping job for {tid}")

            time.sleep(0.02)  # keep the loop snappy

    except KeyboardInterrupt:
        logger.info("[END] Interrupted by user.")
    except Exception:
        logger.error("[FATAL] Unhandled exception in main loop:\n" + traceback.format_exc())

# ──────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Force snapshots for calibration")
    args = parser.parse_args()
    RADAR_TIMEOUT_S = float(config.get("radar_get_timeout", 0.8))
    _init_video_pipelines_for_all()
    if args.calibrate:
        logger.info("[CALIBRATION MODE] Running snapshot capture loop")
        RadarInterface, radar_kwargs = _radar_interface_factory(config)
        radar = RadarInterface(**radar_kwargs)
        ObjectTracker()  # not used but keeps parity
        stall_count = 0
        try:
            while True:
                frame = get_targets_with_timeout(radar, timeout_s=RADAR_TIMEOUT_S)
                if frame is None:
                    stall_count += 1
                    if stall_count >= int(config.get("radar_stall_retries", 3)):
                        logger.warning("[RADAR] timeout ×%d -> reinitializing interface", stall_count)
                        try:
                            radar = RadarInterface(**radar_kwargs)
                        except Exception as e:
                            logger.error(f"[RADAR] reinit failed: {e}")
                        stall_count = 0
                    time.sleep(0.02)
                    continue
                else:
                    stall_count = 0

                targets = frame.get("trackData", []) or []
                if not targets:
                    time.sleep(0.1)
                    continue

                for obj in targets:
                    obj.update({
                        "x": obj.get("posX", 0.0),
                        "y": obj.get("posY", 0.0),
                        "z": obj.get("posZ", 0.0),
                    })

                    cams = config.get("cameras")
                    if isinstance(cams, list) and cams:
                        idx = config.get("selected_camera", 0)
                        cam = cams[min(max(idx, 0), len(cams)-1)]
                    elif isinstance(cams, dict):
                        cam = cams
                    else:
                        cam = {}

                    raw_path = capture_snapshot(
                        camera_url=cam.get("snapshot_url") or cam.get("url") or cam.get("rtsp_url"),
                        username=cam.get("username"),
                        password=cam.get("password")
                    )
                    if raw_path:
                        _ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        temp_name = f"temp_{obj.get('object_id','UNKNOWN')}_{_ts}.jpg"
                        temp_path = os.path.join("snapshots", temp_name)
                        cv2.imwrite(temp_path, cv2.imread(raw_path))
                        logger.info(f"[CALIBRATION] Saved snapshot -> {temp_path}")

                        json_path = temp_path.replace(".jpg", ".json")
                        with open(json_path, "w") as jf:
                            json.dump({
                                "x": float(obj.get("x", 0.0)),
                                "y": float(obj.get("y", 0.0)),
                                "z": float(obj.get("z", 0.0))
                            }, jf, indent=2)
                        logger.info(f"[CALIBRATION] Saved radar JSON -> {json_path}")
                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("[CALIBRATION MODE] Exit requested by user.")
    else:
        main()

def start_main_loop():
    t = Thread(target=main, daemon=True)
    t.start()

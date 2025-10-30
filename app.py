import os
import math
import requests
import time
import json
import uuid
import cv2
import socket
import contextlib
import base64
import zipfile
import email.utils
import shutil
import psutil
import sys
import csv
import logging
import threading
import subprocess
import traceback
import numpy as np
import threading
from io import StringIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import deque, Counter
from io import BytesIO
import io
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from urllib.parse import quote as _urlquote

from flask import (
    Flask, render_template, request, redirect, render_template_string, url_for, send_file, make_response,
    send_from_directory, session, jsonify, flash, abort, Response, stream_with_context, after_this_request
)
from flask_login import (
    LoginManager, login_user, login_required, logout_user,
    UserMixin, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import NotFound
from flask_socketio import SocketIO
from flask_cors import CORS
from requests.auth import HTTPDigestAuth

# External models + tools
import lightgbm as lgb
import re
import psycopg2
import psycopg2.extras
import tempfile
from psycopg2.extras import Json, DictCursor
from psycopg2.pool import SimpleConnectionPool
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse, urlunparse, quote
from tempfile import NamedTemporaryFile
from matplotlib import pyplot as plt
import joblib
import secrets
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Internal modules
from config_utils import load_config, save_config, CONFIG_FILE, db_dsn
from iwr6843_interface import IWR6843Interface, check_radar_connection
from classify_objects import ObjectClassifier
from kalman_filter_tracking import ObjectTracker
from camera import capture_snapshot
from calibration import load_pairs, save_pairs, reset_pairs, add_pair, estimate_intrinsics, fit_extrinsics, publish_model, _paths_for_camera
from bounding_box import annotate_speeding_object
from anpr import run_anpr
from report import generate_pdf_report
from train_lightbgm import fetch_training_data 
from plotter import Live3DPlotter
from kafka_bus import KafkaBus
from main import main as main, start_main_loop

# Flask app init
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config.setdefault("MAX_CONTENT_LENGTH", 64 * 1024 * 1024)  
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
CORS(app)
config = load_config()
kafka_bus = None
try:
    kafka_bus = KafkaBus(config)
except Exception:
    kafka_bus = None
PI_HEALTH_URL = os.environ.get("PI_HEALTH_URL") or (config.get("pi_health_url") if isinstance(config, dict) else None)

log_lock = threading.Lock()
logger = logging.getLogger("RadarApp")
logger.setLevel(logging.DEBUG)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)

# ── Location cache (GPS-first, static fallback) ───────────────────────────────
_loc_cache = {"lat": None, "lon": None, "alt": None, "hdop": None,
              "time": None, "source": None, "ts": 0.0}
_loc_lock = threading.Lock()

# Camera setup
selected = config.get("selected_camera", 0)
cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
camera_url = cam.get("url")
camera_auth = HTTPDigestAuth(cam.get("username"), cam.get("password")) if cam.get("username") else None
camera_frame = None
camera_lock = threading.Lock()
last_frame = None
camera_capture = None
camera_enabled = cam.get("enabled", True)

last_heatmap = None
heatmap_lock = threading.Lock()

def _ffmpeg_bin():
    """Resolve ffmpeg path from env or PATH (Windows-safe); None if not found."""
    try:
        p = os.environ.get("FFMPEG_BIN") or os.environ.get("FFMPEG_PATH")
        if p and os.path.isfile(p):
           return p
    except Exception:
        pass
    try:
        import shutil
        w = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if w:
            return w
    except Exception:
        pass
    # Common Windows installs
    for g in (r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
              r"C:\ffmpeg\bin\ffmpeg.exe"):
        try:
            if os.path.isfile(g):
                return g
        except Exception:
            continue
    return None

# Folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_FOLDER = os.path.join(BASE_DIR, "snapshots")
BACKUP_FOLDER   = os.path.join(BASE_DIR, "backups")
CLIPS_FOLDER    = os.path.join(BASE_DIR, "clips")
BUNDLES_DIR     = os.path.join(BASE_DIR, "violations")   
EXPORTS_DIR     = os.path.join(BASE_DIR, "exports")

# ── PTZ: state + capability helpers (admin-only UI/API) ───────────────────────
# UI and main loop communicate via this small JSON file.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RUNTIME_DIR = os.environ.get("IWR_RUNTIME_DIR", os.path.join(BASE_DIR, "runtime"))
os.makedirs(RUNTIME_DIR, exist_ok=True)
PTZ_STATE_PATH  = os.path.join(RUNTIME_DIR, "ptz_state.json")
PTZ_STATUS_PATH = os.path.join(RUNTIME_DIR, "ptz_status.json")

def _ptz_capable() -> bool:
    """True if a PTZ host is configured or a camera with role='ptz' exists (DB or config)."""
    try:
        live = load_config()
        # Explicit top-level PTZ block wins
        p = (live.get("ptz") or {})
        if p.get("host"):
            return True
        # DB cameras
        try:
            cams, _ = load_cameras_from_db()  # may raise if DB not ready
            for cam in cams or []:
                if str(cam.get("role","")).lower() == "ptz":
                    return True
        except Exception:
            pass
        # Fallback: config cameras list
        for cam in (live.get("cameras") or []):
            if str(cam.get("role","")).lower() == "ptz":
                return True
        return False
    except Exception:
        return False

def _read_ptz_state() -> dict:
    try:
        with open(PTZ_STATE_PATH, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {"enabled": False, "lock_tid": None}

def _read_ptz_status() -> dict:
    """
    Small runtime status JSON the main loop keeps updated.
    Shape: {enabled: bool, locked_by: "auto"|"" , updated_at: epoch}
    Falls back to state when status file isn't present yet.
    """
    try:
        with open(PTZ_STATUS_PATH, "r") as f:
            j = json.load(f) or {}
            j["enabled"] = bool(j.get("enabled", False))
            j["locked_by"] = str(j.get("locked_by") or "")
            return j
    except Exception:
        st = _read_ptz_state()
        return {"enabled": bool(st.get("enabled", False)), "locked_by": ("auto" if st.get("enabled") else ""), "updated_at": 0.0}

def _write_ptz_state(data: dict) -> bool:
    try:
        tmp = PTZ_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"enabled": bool(data.get("enabled", False)),
                       "lock_tid": data.get("lock_tid")}, f)
        os.replace(tmp, PTZ_STATE_PATH)
        return True
    except Exception:
        return False

def _resolve_ptz_cfg_from_env() -> dict:
    """Build a minimal PTZ config dict from config.json or DB cameras."""
    try:
        live = load_config()
    except Exception:
        live = {}
    # 1) explicit top-level block
    p = (live.get("ptz") or {})
    if p.get("host"):
        return p
    # 2) DB camera with role='ptz'
    try:
        cams, _ = load_cameras_from_db()
        for cam in cams or []:
            if str(cam.get("role","")).lower() == "ptz":
                u = cam.get("url") or cam.get("snapshot_url") or ""
                try:
                    pr = urlparse(u)
                    host = pr.hostname or u
                    port = pr.port or 80
                except Exception:
                    host, port = (u, 80)
                return {
                    "host": host,
                    "port": int(port or 80),
                    "username": cam.get("username") or cam.get("user") or "",
                    "password": cam.get("password") or cam.get("pwd") or "",
                    "profile_token": cam.get("profile_token") or ""
                }
    except Exception:
        pass
    # 3) cameras in config.json
    for cam in (live.get("cameras") or []):
        if str(cam.get("role","")).lower() == "ptz":
            u = cam.get("url") or cam.get("snapshot_url") or ""
            try:
                pr = urlparse(u)
                host = pr.hostname or u
                port = pr.port or 80
            except Exception:
                host, port = (u, 80)
            return {
                "host": host,
                "port": int(port or 80),
                "username": cam.get("username") or cam.get("user") or "",
                "password": cam.get("password") or cam.get("pwd") or "",
                "profile_token": cam.get("profile_token") or ""
            }
    return {}

def _ptz_direct_control(job: dict) -> bool:
    """Direct control escape hatch used when the main worker queue is not available."""
    try:
        ptz_cfg = (job.get("ptz_cfg") or {}) if isinstance(job, dict) else {}
        if not (ptz_cfg and ptz_cfg.get("host")):
            ptz_cfg = _resolve_ptz_cfg_from_env()
        if not (ptz_cfg and ptz_cfg.get("host")):
            return False
        try:
            from ptz import PTZController
        except Exception:
            from ptz_controller import PTZController  # type: ignore
        try:
            ctl = PTZController(
                host=ptz_cfg.get("host"),
                username=ptz_cfg.get("username") or ptz_cfg.get("user") or "",
                password=ptz_cfg.get("password") or ptz_cfg.get("pwd") or "",
                port=int(ptz_cfg.get("port") or 80),
                auth_mode=(ptz_cfg.get("auth_mode") or ptz_cfg.get("auth") or "auto"),
                profile_token=str(ptz_cfg.get("profile_token") or "")
            )
        except TypeError:
            # Very old controller signature
            ctl = PTZController(
                host=ptz_cfg.get("host"),
                user=ptz_cfg.get("username") or ptz_cfg.get("user") or "",
                pwd=ptz_cfg.get("password") or ptz_cfg.get("pwd") or ""
            )
        t = job.get("type")
        if t == "nudge":
            vx = float(job.get("vx") or 0.0); vy = float(job.get("vy") or 0.0); vz = float(job.get("vz") or 0.0); sec = float(job.get("sec") or 0.35)
            for name in ("continuous_move","move","nudge","continuousMove","panTilt"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try:
                        return fn(vx=vx, vy=vy, zoom=vz, duration=sec) or True
                    except TypeError:
                        try: return fn(vx, vy, sec, vz) or True
                        except Exception: pass
            return False
        if t == "zoom":
            vz = float(job.get("vz") or 0.0); sec = float(job.get("sec") or 0.35)
            for name in ("continuous_move","zoom","zoom_continuous","zoomMove"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try:
                        return fn(zoom=vz, duration=sec) or True
                    except TypeError:
                        try: return fn(vz, sec) or True
                        except Exception: pass
            if hasattr(ctl, "continuous_move"):
                try:
                    ctl.continuous_move(zoom=vz, duration=sec); return True
                except Exception:
                    pass
            return False
        if t in ("zoom_abs","zoom_to"):
            # Absolute zoom only (no pan/tilt)
            z = job.get("zoom")
            try: z = float(z) if z is not None else None
            except Exception: z = None
            for name in ("zoom_to","absolute_move","absolute"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try:
                        # Prefer dedicated zoom_to if present
                        if name == "zoom_to":
                            fn(z); return True
                        fn(pan=None, tilt=None, zoom=z); return True
                    except TypeError:
                        try: fn(None, None, z); return True
                        except Exception: pass
            return False
        if t == "zoom_auto":
            # Let the controller map distance (meters) -> absolute zoom
            dist = job.get("dist", job.get("distance", None))
            try: dist = float(dist) if dist is not None else None
            except Exception: dist = None
            for name in ("zoom_auto_from_distance","zoomAutoFromDistance"):
                fn = getattr(ctl, name, None)
                if callable(fn) and dist is not None:
                    try: fn(dist); return True
                    except Exception: pass
            # Fallback: linear map (safe defaults) if controller helper missing
            try:
                if dist is None: return False
                dmin, dmax = 4.0, 40.0
                zmin, zmax = 1.0, 10.0
                tlin = 0.0 if dist <= dmin else (1.0 if dist >= dmax else (dmax - dist) / (dmax - dmin))
                z = zmin + tlin * (zmax - zmin)
                # Reuse absolute path
                for name in ("zoom_to","absolute_move","absolute"):
                    fn = getattr(ctl, name, None)
                    if callable(fn):
                        try:
                            if name == "zoom_to":
                                fn(z); return True
                            fn(pan=None, tilt=None, zoom=z); return True
                        except TypeError:
                            try: fn(None, None, z); return True
                            except Exception: pass
            except Exception:
                pass
            return False
        if t == "stop":
            for name in ("stop","Stop","STOP"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try:
                        fn(); return True
                    except Exception:
                        pass
            return False
        if t in ("abs","absolute"):
            pan  = job.get("pan")
            tilt = job.get("tilt")
            zoom = job.get("zoom")
            for name in ("absolute_move","move_absolute","absolute"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try:
                        fn(pan=pan, tilt=tilt, zoom=zoom); return True
                    except TypeError:
                        try: fn(pan, tilt, zoom); return True
                        except Exception: pass
            return False
        if t in ("rel","relative"):
            dpan  = float(job.get("dpan") or 0.0)
            dtilt = float(job.get("dtilt") or 0.0)
            dzoom = float(job.get("dzoom") or 0.0)
            for name in ("relative_move","move_relative","relative"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try:
                        fn(dpan=dpan, dtilt=dtilt, dzoom=dzoom); return True
                    except TypeError:
                        try: fn(dpan, dtilt, dzoom); return True
                        except Exception: pass
            return False
        if t == "preset":
            token = job.get("preset")
            for name in ("goto_preset","preset_goto","gotoPreset"):
                fn = getattr(ctl, name, None)
                if callable(fn):
                    try: fn(token); return True
                    except Exception: pass
            return False
        return False
    except Exception as e:
        logger.debug(f"[PTZ DIRECT] fallback failed: {e}")
        return False

# ── Log sources (whitelist) ───────────────────────────────────────────────────
LOG_SOURCES = {
    # app logs
    "app":              ("system-logs/isk-app.log",                 "App"),
    "app_error":        ("system-logs/isk-app.err.log",             "App Error"),
    "main":             ("system-logs/isk-main.log",                "Main"),

    # system-logs (python scripts now write *.py.log; ps1 wrappers write *.ps1.log)
    "radar":            ("system-logs/radar.log",                   "Radar Service"),
    "activity_monitor": ("system-logs/activity_monitor.log",        "Activity Monitor"),
    "daily_email":      ("system-logs/daily_report_email.ps1.log",   "SMTP Function"),
    "daily_summary":    ("system-logs/daily_summary.log",           "Daily Summary"),
    "db_backup_sync":   ("system-logs/db_backup_and_sync.ps1.log",      "DB Backup & Sync"),
    "db_check":         ("system-logs/db_check.py.log",             "DB Check"),
    "drive_sync":       ("system-logs/drive_sync.log",              "Drive Sync"),
    "health_monitor":   ("system-logs/health_monitor.log",          "Health Monitor"),
    "orphan_cleanup":   ("system-logs/orphan_cleanup.log",          "Orphan Cleanup"),
    "service_watchdog": ("system-logs/service_watchdog.py.log",     "Service Watchdog"),
    "service_restart":  ("system-logs/service_restart.ps1.log",     "Service Restart"),

    # violation-logs
    "violations_csv":   ("radar-logs/violations.csv",               "Violations (CSV)"),
}

def _tail_file(path: str, max_lines: int) -> list[str]:
    from collections import deque
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        dq = deque(f, maxlen=max_lines)
    # return raw lines; client handles trimming
    return list(dq)

#DB
IST = ZoneInfo("Asia/Kolkata")

def to_ist(dt):
    try:
        if dt is None:
            return "N/A"
        if isinstance(dt, str):
            # best-effort parse, leave as-is if it doesn't parse
            try:
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            except Exception:
                return dt
        if getattr(dt, "tzinfo", None) is None:
            # treat DB naive timestamps as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return str(dt)

# Ensure process localtime is IST for any naive strftime users (e.g., reportlib)
os.environ.setdefault("TZ", "Asia/Kolkata")
try:
    time.tzset()
except Exception:
    pass

# ── DB DSN (env → config) ----------------------------------------------------
_env_dsn = os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
DB_DSN = _env_dsn if _env_dsn else db_dsn(config)
if not _env_dsn:
    if "host=localhost" in DB_DSN or "host=127.0.0.1" in DB_DSN:
        if "sslmode" not in DB_DSN:
            DB_DSN += " sslmode=disable"

def _dsn_to_uri(dsn: str) -> str:
    s = (dsn or "").strip()
    if not s:
        return s
    low = s.lower()
    if low.startswith("postgres://"):
        return "postgresql://" + s.split("://", 1)[1]
    if low.startswith("postgresql://"):
        return s
    
    kv = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    from urllib.parse import quote
    host = kv.get("host", "localhost")
    port = kv.get("port", "5432")
    db   = kv.get("dbname") or kv.get("database") or "postgres"
    user = kv.get("user", "")
    pwd  = kv.get("password", "")
    auth = ""
    if user:
        auth = quote(user)
        if pwd:
            auth += ":" + quote(pwd)
        auth += "@"
    return f"postgresql://{auth}{host}:{port}/{quote(db)}"

_pool = None
def _create_pool(dsn: str) -> SimpleConnectionPool:
    return SimpleConnectionPool(
        minconn=1,
        maxconn=8,
        dsn=dsn,
        options="-c timezone=UTC -c client_encoding=UTF8 -c statement_timeout=30000 -c idle_in_transaction_session_timeout=30000",
        application_name="iwr6843isk"
    )

def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        try:
            _pool = _create_pool(DB_DSN)
        except psycopg2.OperationalError as e:
            logger.error(f"[DB] pool init failed: {e}; will retry on next use")
            # Defer raising; first request will retry
            raise
    return _pool

@app.context_processor
def inject_template_helpers():
    return {"time": time, "to_ist": to_ist}

# ---------------- Location utilities ----------------
def _gpsd_try_get_fix(cfg: dict, timeout: float = 1.5):
    """
    Return a dict {lat, lon, alt, hdop, time, source='gps'} if gpsd is available,
    else None. Non-blocking-ish; best-effort.
    """
    try:
        loc_cfg = cfg.get("location", {}) or {}
        host = loc_cfg.get("gpsd_host", "127.0.0.1")
        port = int(loc_cfg.get("gpsd_port", 2947))
        s = socket.create_connection((host, port), timeout=0.8)
        s.sendall(b'?WATCH={"enable":true,"json":true}\n')
        s.settimeout(timeout)
        lat = lon = alt = hdop = ts = None
        end = time.time() + timeout
        buf = b""
        while time.time() < end:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            for raw in buf.decode("utf-8", "ignore").splitlines():
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                cls = msg.get("class")
                if cls == "TPV":
                    lat = msg.get("lat")
                    lon = msg.get("lon")
                    alt = msg.get("alt")
                    ts  = msg.get("time")
                elif cls == "SKY":
                    hdop = msg.get("hdop")
            buf = b""
            if lat and lon:
                s.close()
                return {"lat": lat, "lon": lon, "alt": alt,
                        "hdop": hdop, "time": ts, "source": "gps"}
        s.close()
    except Exception:
        pass
    return None
 
def get_current_location(cfg: dict) -> dict:
    """
    Prefer a recent GPS fix when enabled; otherwise fall back to static site coords.
    """
    site = cfg.get("site", {}) or {}
    loc_cfg = cfg.get("location", {}) or {}
    prefer_gps = bool(loc_cfg.get("prefer_gps", True))
    poll_s = int(loc_cfg.get("poll_seconds", 60))
    now = time.time()

    with _loc_lock:
        # Refresh GPS every poll_s seconds (if preferred)
        if prefer_gps and (now - (_loc_cache.get("ts") or 0)) > poll_s:
            fix = _gpsd_try_get_fix(cfg, timeout=1.5)
            if fix and fix.get("lat") and fix.get("lon"):
                _loc_cache.update({**fix, "ts": now})

        # Choose best available
        if _loc_cache.get("lat") and _loc_cache.get("lon"):
            return {
                "lat": _loc_cache["lat"], "lon": _loc_cache["lon"],
                "alt": _loc_cache.get("alt"), "hdop": _loc_cache.get("hdop"),
                "time": _loc_cache.get("time"),
                "source": _loc_cache.get("source", "gps"),
                "site_id": site.get("site_id"),
                "name": site.get("name"),
                "address": site.get("address")
            }
        # Static fallback (works even without gpsd)
        return {
            "lat": site.get("lat"), "lon": site.get("lon"),
            "alt": None, "hdop": None, "time": None, "source": "static",
            "site_id": site.get("site_id"),
            "name": site.get("name"),
            "address": site.get("address")
        }


# --- User Management ---
class User(UserMixin):
    def __init__(self, id_, username, password_hash, role="viewer"):
        self.id = id_
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def get_id(self):
        return str(self.id)

    @property
    def is_authenticated(self):
        return True

def safe_log(level, msg):
    with log_lock:
        if level == "info":
            logger.info(msg)
        elif level == "error":
            logger.error(msg)
            
def get_user_by_id(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        return User(*row) if row else None


def get_user_by_username(username):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        return User(*row) if row else None

@contextmanager
def get_db_connection():
    """Yield a pooled connection once; never re-yield on exception.
    We retry *acquiring* a connection, but we do not retry the body.
    Broken connections are dropped from the pool.
    """
    conn = None
    # Retry acquiring the connection only (not the body of the with-block)
    for attempt in range(3):
        try:
            pool = _get_pool()
            conn = _pool.getconn()
            break
        except psycopg2.OperationalError as e:
            logger.warning(f"[DB CONNECT RETRY] Attempt {attempt+1} failed: {e}")
            time.sleep(0.2 * (2**attempt))
    if conn is None:
        # All attempts failed
        raise psycopg2.OperationalError("Could not obtain a database connection from pool")

    broken = False
    try:
        yield conn
    except psycopg2.OperationalError as e:
        broken = True
        logger.warning(f"[DB OPERATIONAL ERROR] {e}")
        raise
    except Exception as e:
        # treat as broken to be safe
        broken = True
        logger.error(f"[DB ERROR] Unexpected: {e}")
        raise
    finally:
        try:
            if conn is not None:
                # drop broken connections
                try:
                    pool = _get_pool()
                    pool.putconn(conn, close=broken)
                except Exception:
                    # if pool itself errored, close the conn directly
                    conn.close()
        except Exception:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

def _ensure_model_metadata_schema():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id SERIAL PRIMARY KEY,
                version TEXT NOT NULL,
                features JSONB,
                accuracy DOUBLE PRECISION,
                labels JSONB,
                trained_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()

def _ensure_model_info_schema():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_info (
                id SERIAL PRIMARY KEY,
                accuracy DOUBLE PRECISION,
                method   TEXT,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()

def _to_py_scalar(x):
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    return x if isinstance(x, (str, int, float, bool)) else str(x)

def _jsonify_list(seq):
    if not seq:
        return []
    return [_to_py_scalar(v) for v in seq]

def save_model_metadata_db(version, features, accuracy, labels):
    """Authoritative metadata table: version, features, accuracy, labels."""
    _ensure_model_metadata_schema()
    # make lists JSON-serializable
    features = _jsonify_list(features)
    labels   = _jsonify_list(labels)
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_metadata (version, features, accuracy, labels)
            VALUES (%s, %s, %s, %s)
        """, (version, Json(features), accuracy, Json(labels)))
        conn.commit()

def get_latest_model_metadata_db():
    _ensure_model_metadata_schema()
    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM model_metadata ORDER BY trained_at DESC LIMIT 1")
        return cur.fetchone()

def _next_model_version():
    _ensure_model_metadata_schema()
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT version FROM model_metadata ORDER BY trained_at DESC LIMIT 1")
        row = cur.fetchone()
    if row and row[0]:
        m = re.match(r"^v(\d+)$", str(row[0]).strip())
        if m:
            return f"v{int(m.group(1)) + 1}"
    return "v1"

def _viz_emit(kind: str, jpg_bytes: bytes) -> None:
    try:
        socketio.emit(
            "viz_frame",
            {"kind": kind, "jpg_b64": base64.b64encode(jpg_bytes).decode("ascii"), "ts": time.time()},
            broadcast=True,
        )
        # Also publish to Kafka for cross-process dashboards
        try:
            if kafka_bus and bool(config.get("kafka", {}).get("enabled")):
                kafka_bus.publish(
                    topic=str(config.get("kafka", {}).get("topic_plotter", "plotter_frames")),
                    value={
                        "kind": kind,
                        "jpg_b64": base64.b64encode(jpg_bytes).decode("ascii"),
                        "ts": time.time(),
                    },
                    key=kind,
                )
        except Exception:
            pass
    except Exception:
        # keep server robust even if no clients are connected
        pass

plotter = Live3DPlotter(frame_sink=_viz_emit)
# ---- CAM Helpers ----

def _embed_basic_in_rtsp(url: str, username: str | None, password: str | None) -> str:
    """
    Return a credentials-safe RTSP URL.
    - If explicit username/password are provided, embed them (percent-encoded).
    - Else, reuse any creds already present in the URL but percent-encode them.
    """
    try:
        if not url or not url.lower().startswith("rtsp://"):
            return url
        p = urlparse(url)
        user = username if (username is not None and username != "") else (p.username or "")
        pwd  = password if (password is not None and password != "") else (p.password or "")
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        auth = f"{quote(user, safe='')}:{quote(pwd, safe='')}@" if (user or pwd) else ""
        netloc = f"{auth}{host}{port}"
        return urlunparse((p.scheme, netloc, p.path or "", p.params or "", p.query or "", p.fragment or ""))
    except Exception:
        return url

def _si_no_window():
    # On Windows, prevent a console window blip for child processes
    try:
        import subprocess as _sp
        si = _sp.STARTUPINFO()
        si.dwFlags |= _sp.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        return si
    except Exception:
        return None

def _cf_no_window():
    # CREATE_NO_WINDOW
    return 0x08000000 if os.name == "nt" else 0

def _probe_camera(cam: dict, timeout_s: float = 5.0) -> bool:
    """
    Liveness probe for one camera.
    - RTSP: ffmpeg -t 1 -f null -
    - MJPEG: read a chunk and look for a full JPEG
    - snapshot: fetch once, check JPEG SOI marker
    """
    try:
        return _grab_one_jpeg(cam, timeout_s=timeout_s) is not None
    except Exception as e:
        logger.warning(f"[CAMERA TEST] {cam.get('name') or cam.get('id')} failed: {e}")
        return False

def _grab_one_jpeg(cam: dict, timeout_s: float = 6.0) -> bytes|None:
    """
    Return a single JPEG frame from cam, without persisting.
    - snapshot: direct GET
    - rtsp: ffmpeg vframes=1 to a temp file
    - mjpeg: read until a full JPEG appears
    """
    url = (cam.get("url") or cam.get("snapshot_url") or "").strip()
    u = cam.get("username") or ""
    p = cam.get("password") or ""
    st = str(cam.get("stream_type", "mjpeg")).lower()
    try:
        if st == "snapshot":
            r = requests.get(url, auth=HTTPDigestAuth(u, p) if (u and p) else None,
                             timeout=timeout_s)
            try:
                if r.status_code == 200 and r.content.startswith(b"\xff\xd8"):
                    return r.content
            finally:
                try: r.close()
                except: pass
            return None
        if st == "mjpeg":
            r = requests.get(url, auth=HTTPDigestAuth(u, p) if (u and p) else None,
                             stream=True, timeout=timeout_s)
            try:
                buf = b""
                for chunk in r.iter_content(4096):
                    buf += chunk
                    si = buf.find(b"\xff\xd8")
                    ei = buf.find(b"\xff\xd9")
                    if si != -1 and ei != -1 and ei > si:
                        return buf[si:ei+2]
            finally:
                try: r.close()
                except: pass
            return None

        # RTSP single frame via FFmpeg; if not available, fall back to HTTP snapshot (if provided)
        rtsp = _embed_basic_in_rtsp(url, u, p)
        ffmpeg_bin = _ffmpeg_bin()
        if not ffmpeg_bin:
            # Fallback to snapshot_url if available
            snap = (cam.get("snapshot_url") or "").strip()
            if snap:
                try:
                    r = requests.get(snap, auth=HTTPDigestAuth(u, p) if (u and p) else None, timeout=timeout_s)
                    try:
                        if r.status_code == 200 and r.content.startswith(b"\xff\xd8"):
                            return r.content
                    finally:
                        try: r.close()
                        except: pass
                except Exception:
                    pass
            return None
        # FFmpeg path (Windows-safe temp handling)
        tmp = NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        try:
            tmp.close()
            cmd = [ffmpeg_bin, "-y", "-rtsp_transport", "tcp",
                   "-stimeout", str(int(timeout_s * 1_000_000)),
                   "-i", rtsp, "-vframes", "1", "-q:v", "2", tmp_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=max(8, int(timeout_s)+2), check=False)
            try:
                with open(tmp_path, "rb") as f:
                    data = f.read()
                if data.startswith(b"\xff\xd8"):
                    return data
            except Exception:
                pass
            # As a secondary fallback, try snapshot_url if present
            snap = (cam.get("snapshot_url") or "").strip()
            if snap:
                try:
                    r = requests.get(snap, auth=HTTPDigestAuth(u, p) if (u and p) else None, timeout=timeout_s)
                    try:
                        if r.status_code == 200 and r.content.startswith(b"\xff\xd8"):
                            return r.content
                    finally:
                        try: r.close()
                        except: pass
                except Exception:
                    pass
            return None
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    except Exception as e:
        logger.debug(f"[FRAME GRAB] {cam.get('name')} error: {e}")
        return None

def save_cameras_to_db(cameras, selected_idx=None):
    """
    Upsert cameras by ID (stable IDs), and delete any rows not present in the payload.
    Each camera dict may include: id, name, url, snapshot_url, username, password,
    stream_type, role, enabled (-> is_active).
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        # Snapshot existing IDs
        cur.execute("SELECT id FROM cameras")
        existing_ids = {r[0] for r in cur.fetchall()}
        kept_ids = set()

        for cam in cameras or []:
            cid = cam.get("id")
            name = cam.get("name")
            url = cam.get("url")
            snap = cam.get("snapshot_url")
            user = cam.get("username")
            pwd  = cam.get("password")
            st   = (cam.get("stream_type") or "mjpeg").lower()
            role = (cam.get("role") or "").lower()
            # Accept the three supported roles; default to 'aux' only if invalid/missing
            if role not in ("primary", "aux", "ptz"):
                role = "aux"
            active = bool(cam.get("enabled", True))

            if cid:  # UPDATE existing (or insert-with-id if missing)
                cur.execute("""
                    INSERT INTO cameras (id, name, url, snapshot_url, username, password,
                                         stream_type, role, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name=EXCLUDED.name,
                        url=EXCLUDED.url,
                        snapshot_url=EXCLUDED.snapshot_url,
                        username=EXCLUDED.username,
                        password=EXCLUDED.password,
                        stream_type=EXCLUDED.stream_type,
                        role=EXCLUDED.role,
                        is_active=EXCLUDED.is_active
                """, (int(cid), name, url, snap, user, pwd, st, role, active))
                kept_ids.add(int(cid))
            else:    # INSERT new → return id
                cur.execute("""
                    INSERT INTO cameras (name, url, snapshot_url, username, password,
                                         stream_type, role, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (name, url, snap, user, pwd, st, role, active))
                new_id = cur.fetchone()[0]
                kept_ids.add(int(new_id))

        # Delete rows that were removed from the form/payload
        to_delete = list(existing_ids - kept_ids)
        if to_delete:
            cur.execute("DELETE FROM cameras WHERE id = ANY(%s)", (to_delete,))
        conn.commit()

def _resolve_cam_and_id(raw_sel, cams):
    """
    Resolve a camera selection (DB id or list index) to (camera_dict, db_id).
    raw_sel can be str/int; cams is the list returned by load_cameras_from_db().
    """
    cam = {}
    sel_id = None
    try:
        sel_id = int(raw_sel)
    except Exception:
        sel_id = None
    # 1) Try direct match on DB id
    for c in (cams or []):
        try:
            if sel_id is not None and int(c.get("id")) == sel_id:
                return c, int(c["id"])
        except Exception:
            pass
    # 2) Fall back to treating raw_sel as list index
    try:
        idx = int(raw_sel if raw_sel is not None else 0)
        if 0 <= idx < len(cams or []):
            cam = cams[idx]
            return cam, int(cam.get("id"))
    except Exception:
        pass
    # 3) No match → first active cam (or None, None)
    if cams:
        return cams[0], int(cams[0].get("id"))
    return {}, None

def load_cameras_from_db():
    """
    Returns (cameras_list, selected_idx). Supports multiple active cameras.
    """
    cams = []
    # Ensure table exists and the CHECK constraint allows all roles we support
    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        try:
            # Create table with correct CHECK if missing
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    url TEXT,
                    snapshot_url TEXT,
                    username TEXT,
                    password TEXT,
                    stream_type TEXT DEFAULT 'mjpeg',
                    role TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            conn.commit()
        except Exception:
            conn.rollback()
        # Align/replace the CHECK constraint to allow 'primary','aux','ptz'
        try:
            cur.execute("ALTER TABLE cameras DROP CONSTRAINT IF EXISTS cameras_role_check")
            cur.execute("""
                ALTER TABLE cameras
                ADD CONSTRAINT cameras_role_check
                CHECK (role IN ('primary','aux','ptz'))
            """)
            conn.commit()
        except Exception:
            conn.rollback()
        cur.execute("""SELECT id, name, url, snapshot_url, username, password, stream_type, role, is_active
                       FROM cameras ORDER BY id ASC""")
        for r in cur.fetchall():
            cams.append({
                "id": int(r["id"]),
                "name": r["name"] or f"Camera {r['id']}",
                "url": r["url"],
                "snapshot_url": r["snapshot_url"],
                "username": r["username"],
                "password": r["password"],
                "stream_type": (r["stream_type"] or "mjpeg").lower(),
                "role": (r["role"] or "").lower(),
                "enabled": bool(r["is_active"]),
            })
    sel = int(config.get("selected_camera", 0))
    sel = max(0, min(sel, len(cams)-1)) if cams else 0
    return cams, sel

def set_camera_active_state(cam_id: int, active: bool) -> None:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE cameras SET is_active=%s WHERE id=%s", (bool(active), int(cam_id)))
        conn.commit()

def _active_cameras_from_db():
    cams, _ = load_cameras_from_db()
    return [c for c in cams if c.get("enabled", True)]

BOUNDARY = "frame"

def _camera_by_id(cam_id: int) -> dict | None:
    cams, _ = load_cameras_from_db()
    for c in cams:
        if int(c.get("id", -1)) == int(cam_id):
            return c
    return None

def _jpeg_generator_for_cam(cam: dict, fps: float = 3.0):
    """
    Yields multipart MJPEG for the given camera at a capped FPS.
    Works for snapshot/MJPEG directly; for RTSP we grab single frames via ffmpeg.
    """
    period = max(0.2, 1.0 / float(fps))
    while True:
        if not cam or not cam.get("enabled", True):
            time.sleep(0.5)
            continue
        data = _grab_one_jpeg(cam, timeout_s=6.0)  # uses RTSP/snapshot/MJPEG logic
        if data:
            # best-effort publish a lightweight frame to Kafka (base64 JPG)
            try:
                if kafka_bus and bool(config.get("kafka", {}).get("enabled")):
                    cam_name = str(cam.get("name") or cam.get("id") or "camera")
                    kafka_bus.publish(
                        topic=str(config.get("kafka", {}).get("topic_frames", "camera_frames")),
                        value={
                            "camera": cam_name,
                            "jpg_b64": base64.b64encode(data).decode("ascii"),
                            "ts": time.time(),
                        },
                        key=cam_name,
                    )
            except Exception:
                pass
            yield (b"--" + BOUNDARY.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
                   data + b"\r\n")
        time.sleep(period)

def _row_by_snapshot_basename(conn, basename):
    """Fetch a radar_data row by the basename of snapshot_path (latest first)."""
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT
                measured_at, datetime, sensor, object_id, type, confidence, speed_kmh,
                velocity, distance, direction, motion_state, snapshot_path, snapshot_type,
                azimuth, elevation, clip_path, clip_status, clip_duration_s, clip_fps,
                clip_size_bytes, clip_sha256, clip_gdrive_link
            FROM radar_data
            WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
              AND regexp_replace(snapshot_path, '^.*[\\\\/]', '') = %s
            ORDER BY measured_at DESC
            LIMIT 1
        """, (basename,))
        return cur.fetchone()

def _is_clip_ready(row):
    """True if DB says ready and file exists."""
    try:
        return (row and str(row.get("clip_status") or "").lower() == "ready" 
                and row.get("clip_path") 
                and os.path.exists(row["clip_path"]))
    except Exception:
        return False

def _guess_bundle_from_snapshot_path(snap_path: str) -> str | None:
    """
    Resolve the bundle directory for a given snapshot_path.
    If snapshots/<name>.jpg is a symlink into violations/.../image.jpg, we return that dir.
    Else, try to infer from the basename pattern: <prefix>_<oid>_<YYYYMMDD_HHMMSS(_micro)?>.jpg
    """
    try:
        if not snap_path:
            return None
        real = os.path.realpath(snap_path)
        # Symlink aware: image.jpg lives directly inside the bundle dir
        if os.path.basename(real) == "image.jpg":
            bdir = os.path.dirname(real)
            return bdir if os.path.isdir(bdir) else None
        # Fallback: infer from basename
        base = os.path.splitext(os.path.basename(snap_path))[0]
        parts = base.split("_")
        if len(parts) >= 3:
            prefix = parts[0]                    # violation / calib
            oid    = parts[1]
            ts     = "_".join(parts[2:])         # YYYYMMDD_HHMMSS[_micro]
            date_key = ts.split("_")[0]          # YYYYMMDD
            cand = os.path.join(BUNDLES_DIR, date_key, f"{prefix}_{oid}_{ts}")
            if os.path.isdir(cand):
                return cand
        return None
    except Exception:
        return None

def _resolve_bundle_by_basename(basename: str) -> str | None:
    """
    Resolve the bundle directory for a given snapshot basename.
    Works for both:
      - legacy: DB row where snapshot_path points anywhere
      - new alias: /snapshots/<bundle_dir_name>.jpg -> violations/.../image.jpg
    """
    try:
        if not basename:
            return None

        # 1) New alias path: /snapshots/<alias>.jpg symlinks to violations/.../image.jpg
        alias_fp = os.path.join(SNAPSHOT_FOLDER, basename)
        if os.path.exists(alias_fp):
            try:
                real = os.path.realpath(alias_fp)
                if os.path.basename(real) == "image.jpg":
                    bdir = os.path.dirname(real)
                    if os.path.isdir(bdir):
                        return bdir
            except Exception:
                pass

        # 2) Legacy fallback via DB lookup by original snapshot basename
        with get_db_connection() as conn:
            row = _row_by_snapshot_basename(conn, basename)
        snap_path = (row or {}).get("snapshot_path")
        return _guess_bundle_from_snapshot_path(snap_path or "")
    except Exception:
        return None

def _bundle_manifest(bundle_dir: str) -> dict:
    """Return manifest for UI/API.
    - image: canonical main image (image.jpg if present)
    - clip:  canonical main clip  (clip.mp4 if present)
    - meta:  meta.json if present
    - images[]: all image files in bundle (JPG/PNG), basenames
    - videos[]: all video files in bundle (MP4/MOV)"""
    try:
        img  = os.path.join(bundle_dir, "image.jpg")
        mp4  = os.path.join(bundle_dir, "clip.mp4")
        meta = os.path.join(bundle_dir, "meta.json")
        images = []
        videos = []
        for root, _, files in os.walk(bundle_dir):
            for f in files:
                lf = f.lower()
                rel = os.path.relpath(os.path.join(root, f), bundle_dir)  # preserve subpaths
                if lf.endswith((".jpg", ".jpeg", ".png")):
                    images.append(rel)
                if lf.endswith((".mp4", ".mov")):
                    videos.append(rel)
        return {
            "exists": True,
            "dir": bundle_dir,
            "image": (os.path.basename(img) if os.path.exists(img) else None),
            "image_size": (os.path.getsize(img) if os.path.exists(img) else None),
            "clip": (os.path.basename(mp4) if os.path.exists(mp4) else None),
            "clip_size": (os.path.getsize(mp4) if os.path.exists(mp4) else None),
            "meta": (os.path.basename(meta) if os.path.exists(meta) else None),
            "images": sorted(list(set(images))),
            "videos": sorted(list(set(videos))),
        }
    except Exception:
        return {"exists": False, "dir": bundle_dir}

# ---- Shared helpers for guided calibration -----------------------------------

def _calib_paths():
    basedir = os.path.dirname(os.path.abspath(__file__))
    caldir = os.path.join(basedir, "calibration")
    live_path = os.path.join(caldir, "camera_model.json")
    staged_path = os.path.join(caldir, "camera_model_staging.json")
    pairs_path = os.path.join(caldir, "calibration_pairs.json")
    os.makedirs(caldir, exist_ok=True)
    return caldir, live_path, staged_path, pairs_path
    
def _live_json_paths() -> list[str]:
    """
    Candidate locations for the main loop's live radar objects JSON.
    Order matters; we pick the freshest existing file.
    """
    basedir = os.path.dirname(os.path.abspath(__file__))
    # Allow ~ expansion for env-provided paths
    envp = os.path.expanduser(os.environ.get("IWR6843ISK_LIVE_DIR", "") or "")
    envf = os.path.expanduser(os.environ.get("IWR6843ISK_LIVE_JSON", "") or "")
    try:
        uid = str(os.getuid())  # POSIX
    except AttributeError:
        # Windows fallback: prefer USERNAME/USER, else "0"
        uid = str(
            os.environ.get("USERNAME")
            or os.environ.get("USER")
            or "0"
        )
    out = []
    if envf:
        out.append(envf)
    if envp:
        out.append(os.path.join(envp, "live_objects.json"))
    out += [
        os.path.join(basedir, "live_objects.json"),
        os.path.join(os.path.expanduser("~"), "iwr6843isk", "live_objects.json"),
        f"/run/user/{uid}/iwr6843isk/live_objects.json",
        "/run/iwr6843isk/live_objects.json",
        "/tmp/iwr6843isk/live_objects.json",
    ]
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        tmp   = os.environ.get("TEMP") or os.environ.get("TMP")
        if local:
            out.append(os.path.join(local, "iwr6843isk", "live_objects.json"))
        if tmp:
            out.append(os.path.join(tmp, "iwr6843isk", "live_objects.json"))
    seen=set(); ordered=[]
    for p in out:
        if p and p not in seen:
            ordered.append(p); seen.add(p)
    return ordered

def _load_latest_candidates(max_age_s: float = 2.0) -> list[dict]:
    """
    Read the freshest live_objects.json and return a canonicalized list of
    candidates with keys: distance_m, azimuth_deg, elevation_deg.
    Falls back to the running radar pipeline if the file is stale/missing.
    """
    def _canon_from_obj(o: dict):
        """Normalize various object shapes into {distance_m, azimuth_deg, elevation_deg}."""
        try:
            # Already canonical
            if all(k in o for k in ("distance_m", "azimuth_deg", "elevation_deg")):
                return {
                    "distance_m": float(o["distance_m"]),
                    "azimuth_deg": float(o["azimuth_deg"]),
                    "elevation_deg": float(o["elevation_deg"]),
                }
            # Native polar
            if "distance" in o and "azimuth" in o:
                return {
                    "distance_m": float(o.get("distance", 0.0)),
                    "azimuth_deg": float(o.get("azimuth", 0.0)),
                    "elevation_deg": float(o.get("elevation", 0.0)),
                }
            # Cartesian
            if all(k in o for k in ("x", "y", "z")):
                x, y, z = float(o["x"]), float(o["y"]), float(o["z"])
                r  = (x*x + y*y + z*z) ** 0.5
                az = math.degrees(math.atan2(x, y))
                el = math.degrees(math.atan2(z, (x*x + y*y) ** 0.5))
                return {"distance_m": r, "azimuth_deg": az, "elevation_deg": el}
        except Exception:
            pass
        return None

    def _radar_fallback():
        try:
            objs = []
            if 'radar' in globals():
                # Support either interface shape, without raising AttributeError
                if hasattr(radar, "get_targets"):
                    objs = radar.get_targets() or []
                elif hasattr(radar, "get_latest_frame"):
                    fr = radar.get_latest_frame() or {}
                    objs = fr.get("objects", []) or []
            out = []
            for o in list(objs):
                c = _canon_from_obj(o)
                if c:
                    out.append(c)
            return out
        except Exception as e:
            logger.debug(f"[CALIB] radar fallback failed: {e}")
            return []

    # Pick freshest candidate file
    best = None
    best_mtime = -1
    for p in _live_json_paths():
        try:
            if os.path.exists(p):
                mt = os.path.getmtime(p)
                if mt > best_mtime:
                    best, best_mtime = p, mt
        except Exception:
            continue

    if not best:
        return _radar_fallback()

    now = time.time()
    try:
        with open(best, "r") as f:
            j = json.load(f) or {}
        # prefer embedded timestamp, else file mtime
        t = float(j.get("t", best_mtime))
        if (now - t) > max_age_s:
            return _radar_fallback()

        objs = list(j.get("objects", []))
        out = []
        for o in objs:
            c = _canon_from_obj(o)
            if c:
                out.append(c)
        return out or _radar_fallback()
    except Exception:
        return _radar_fallback()

# --- Deletion helpers --------------------------------------------------------
def _safe_unlink(path: str) -> bool:
    """Best-effort file delete."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            return True
    except Exception as e:
        try:
            logger.warning(f"[DELETE FILE] {path}: {e}")
        except Exception:
            pass
    return False

def _delete_clip_files(row: dict) -> dict:
    """
    Delete local clip (.mp4) and its sidecar (.json) if present.
    Returns a dict with the booleans of what was deleted.
    """
    deleted = {"mp4": False, "json": False}
    try:
        clip_fp = (row or {}).get("clip_path") or ""
        clip_fp = clip_fp.strip()
        if clip_fp:
            deleted["mp4"] = _safe_unlink(clip_fp)
            base, _ = os.path.splitext(clip_fp)
            deleted["json"] = _safe_unlink(base + ".json")
    except Exception:
        pass
    return deleted

def _delete_files_for_rows(rows):
    """
    Best-effort removal of on-disk artifacts for each DB row:
    - snapshot image
    - plate crop
    - local clip + sidecar json
    - violations bundle files (image.jpg / clip.mp4 / meta.json) if resolvable
    """
    deleted_files = 0
    for r in rows or []:
        try:
            snap = (r.get("snapshot_path") or "").strip()
            if snap and os.path.exists(snap):
                if _safe_unlink(snap): deleted_files += 1
            plate = (r.get("plate_crop_path") or "").strip()
            if plate and os.path.exists(plate):
                if _safe_unlink(plate): deleted_files += 1
            # clip + sidecar
            d = _delete_clip_files(r)
            deleted_files += int(d.get("mp4", False)) + int(d.get("json", False))
            # try bundle dir (if snapshot is symlinked there or pattern-resolvable)
            bdir = _guess_bundle_from_snapshot_path(snap) if snap else None
            if bdir and os.path.isdir(bdir):
                for name in ("image.jpg", "clip.mp4", "meta.json"):
                    p = os.path.join(bdir, name)
                    if _safe_unlink(p): deleted_files += 1
                # remove dir if empty
                try:
                    if not os.listdir(bdir):
                        os.rmdir(bdir)
                except Exception:
                    pass
        except Exception:
            pass
    return deleted_files

# --- MP4 validity + repair helpers (best-effort, lightweight) -----------------
def _looks_like_mp4(path: str) -> bool:
    """
    Very light sanity check: MP4 brand + moov discoverable early.
    Returns True only if the file *appears* to be web-playable already.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 40_000:
            return False
        with open(path, "rb") as f:
            head = f.read(1024 * 256)  # peek first 256 KiB for atoms
        if b"ftyp" not in head:
            return False
        # If moov is not in the head window, browsers may choke (moov at tail).
        return (b"moov" in head)
    except Exception:
        return False

def _ensure_public_snapshot_filename(snap_path: str) -> str:
    """
    Return a filename that is guaranteed to be servable from SNAPSHOT_FOLDER.
    - If snap_path is already a legacy snapshots/<unique>.jpg → ensure a file/symlink exists and return that basename.
    - If snap_path points inside a violations bundle (…/image.jpg) → create a unique alias in snapshots/ that
      symlinks to the bundle image and return that alias (e.g., <bundle_dir_name>.jpg).
    """
    try:
        if not snap_path:
            return "no_image.jpg"
        base = os.path.basename(snap_path)
        # Case 1: legacy unique name under snapshots/
        if base != "image.jpg":
            link = os.path.join(SNAPSHOT_FOLDER, base)
            if not os.path.exists(link) and os.path.exists(snap_path):
                try:
                    rel = os.path.relpath(snap_path, start=SNAPSHOT_FOLDER)
                    os.symlink(rel, link)
                except Exception:
                    pass
            return base
        # Case 2: bundle path → alias to bundle/image.jpg as <bundle_dir_name>.jpg
        bdir = _guess_bundle_from_snapshot_path(snap_path)
        if not bdir:
            return base
        alias = os.path.basename(bdir) + ".jpg"
        link = os.path.join(SNAPSHOT_FOLDER, alias)
        target = os.path.join(bdir, "image.jpg")
        if not os.path.exists(link) and os.path.exists(target):
            try:
                rel = os.path.relpath(target, start=SNAPSHOT_FOLDER)
                os.symlink(rel, link)
            except Exception:
                pass
        return alias
    except Exception:
        return "no_image.jpg"

def _ensure_web_playable_mp4(path: str) -> None:
    """
    If file isn't clearly faststart MP4, re-mux with +faststart and even dims.
    Best-effort; silently returns on any failure.
    """
    try:
        if not os.path.exists(path):
            return
        if _looks_like_mp4(path):
            return  # already looks browser-friendly
        import subprocess
        ff = _ffmpeg_bin()
        if not ff:
            return
        tmp = path + ".fixed.mp4"  # ensure ffmpeg knows the container
        # 1) Lightweight: remux with moov at head
        cmd_copy = [
            ff, "-y", "-loglevel", "error",
            "-i", path,
            "-c", "copy",
            "-movflags", "+faststart",
            "-f", "mp4",
            tmp
        ]
        res = subprocess.run(cmd_copy)
        if res.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 40_000:
            # 2) Fallback: re-encode to even dims + yuv420p
            cmd_re = [
                ff, "-y", "-loglevel", "error",
                "-i", path,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-f", "mp4",
                tmp
            ]
            subprocess.run(cmd_re, check=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) >= 40_000:
            os.replace(tmp, path)  # atomic swap
        else:
            try: os.remove(tmp)
            except Exception: pass
    except Exception:
        # best-effort by design
        pass

def _ensure_streamable_video(path: str) -> tuple[str, str]:
    """
    Return (serve_path, mimetype) guaranteeing browser-playable content.
    - For .mp4: ensure +faststart and return as video/mp4.
    - For .mov/.m4v: remux to adjacent <stem>.stream.mp4 (faststart) when ffmpeg is available,
      else fall back to serving quicktime.
    """
    try:
        lp = path.lower()
        if lp.endswith(".mp4"):
            try:
                _ensure_web_playable_mp4(path)
            except Exception:
                pass
            return path, "video/mp4"

        if lp.endswith((".mov", ".m4v")):
            # sidecar streamable mp4 next to MOV/M4V
            sidecar = os.path.splitext(path)[0] + ".stream.mp4"
            try:
                import subprocess
                ff = _ffmpeg_bin()
                if not ff:
                    # fallback: let the browser try quicktime
                    return path, "video/quicktime"

                # regenerate if missing, tiny, or stale vs source
                src_mtime = os.path.getmtime(path)
                needs = True
                if os.path.exists(sidecar):
                    try:
                        needs = (os.path.getmtime(sidecar) < src_mtime) or (os.path.getsize(sidecar) < 40_000)
                    except Exception:
                        needs = True
                if needs:
                    tmp = sidecar + ".tmp"
                    rc = subprocess.run(
                        [ff, "-y","-loglevel","error","-i",path,"-c","copy","-movflags","+faststart","-f","mp4",tmp]
                    ).returncode
                    if rc != 0 or (not os.path.exists(tmp)) or (os.path.getsize(tmp) < 40_000):
                        subprocess.run(
                            [ff, "-y","-loglevel","error","-i",path,
                             "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",
                             "-c:v","libx264","-preset","veryfast","-crf","23",
                             "-pix_fmt","yuv420p","-movflags","+faststart","-f","mp4",tmp],
                            check=True
                        )
                    os.replace(tmp, sidecar)
                try:
                    _ensure_web_playable_mp4(sidecar)
                except Exception:
                    pass
                return sidecar, "video/mp4"
            except Exception:
                return path, "video/quicktime"

        return path, "application/octet-stream"
    except Exception:
        return path, "application/octet-stream"

def _update_clip_fields_by_basename(conn, basename, **fields):
    """Update clip_* columns by snapshot basename (latest row)."""
    if not fields:
        return 0
    cols = []
    vals = []
    for k, v in fields.items():
        cols.append(f"{k} = %s")
        vals.append(v)
    vals.append(basename)
    with conn.cursor() as cur:
        # Postgres-compatible single-row UPDATE using a CTE and ctid
        cur.execute(f"""
            WITH cand AS (
              SELECT ctid
              FROM radar_data
              WHERE snapshot_path IS NOT NULL
                AND snapshot_path <> ''
                AND regexp_replace(snapshot_path, '^.*[\\\/]', '') = %s
              ORDER BY measured_at DESC
              LIMIT 1
            )
            UPDATE radar_data r
            SET {", ".join(cols)}
            FROM cand
            WHERE r.ctid = cand.ctid
        """, vals)
        conn.commit()
        return cur.rowcount

def _rclone_copy_and_link(local_path: str, remote: str = None, remote_dir: str = None):
    """
    Copy clip to Google Drive via rclone and return a shareable link if supported.
    Requires a configured remote (default: env RCLONE_REMOTE or 'gdrive').
    Destination defaults to 'RadarClips' folder at remote root (env RCLONE_CLIP_DIR).
    """
    remote = remote or os.environ.get("RCLONE_REMOTE", "gdrive")
    remote_dir = remote_dir or os.environ.get("RCLONE_CLIP_DIR", "RadarClips")
    if not shutil.which("rclone"):
        return None
    try:
        dest = f"{remote}:{remote_dir}"
        # Ensure folder exists (best-effort)
        subprocess.run(["rclone", "mkdir", dest], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Copy file
        subprocess.run(["rclone", "copy", "--drive-chunk-size", "64M", local_path, dest], check=True)
        # Try to fetch a public link (if the remote supports it)
        base = os.path.basename(local_path)
        link_target = f"{dest}/{base}"
        out = subprocess.run(["rclone", "link", link_target], check=False, capture_output=True, text=True)
        url = (out.stdout or "").strip()
        return url if url.startswith("http") else None
    except Exception as e:
        logger.warning(f"[RCLONE] Upload/link failed: {e}")
        return None

def update_user_activity(user_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_activity (user_id, last_activity)
            VALUES (%s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET last_activity = EXCLUDED.last_activity
        """, (user_id,))
        conn.commit()

def get_active_users(minutes=30):
    """Return active users in the last `minutes` using DB time (timezone-safe)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT u.id, u.username, u.role, ua.last_activity
                FROM users u
                JOIN user_activity ua ON u.id = ua.user_id
                WHERE ua.last_activity >= NOW() - (%s || ' minutes')::INTERVAL
                ORDER BY ua.last_activity DESC
            """, (str(minutes),))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"[USER ACTIVITY] Query failed: {e}")
        return []

def clean_inactive_sessions():
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_activity WHERE last_activity < %s", (cutoff,))
            conn.commit()
    except Exception as e:
        logger.error(f"[SESSION CLEANUP] Failed: {e}")

def is_admin():
    return current_user.is_authenticated and getattr(current_user, "role", None) == "admin"

def apply_pagination(total_items, page=1, limit=100):
    total_pages = max((total_items + limit - 1) // limit, 1)
    current_page = max(min(page, total_pages), 1)
    offset = (current_page - 1) * limit
    return offset, limit, total_pages, current_page

def ensure_directories():
    """Ensure required folders exist"""
    for directory in [SNAPSHOT_FOLDER, BACKUP_FOLDER, CLIPS_FOLDER, BUNDLES_DIR]:
        os.makedirs(directory, exist_ok=True)

def validate_snapshots():
    """Remove missing snapshot paths from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT id, snapshot_path FROM radar_data WHERE snapshot_path IS NOT NULL")
            rows = cursor.fetchall()

            invalid_count = 0
            for row in rows:
                path = row['snapshot_path']
                if not os.path.exists(path):
                    cursor.execute("UPDATE radar_data SET snapshot_path = NULL WHERE id = %s", (row['id'],))
                    invalid_count += 1
            conn.commit()
            logger.info(f"[SNAPSHOT VALIDATOR] Removed {invalid_count} broken paths")
            return invalid_count
    except Exception as e:
        logger.error(f"[SNAPSHOT VALIDATOR ERROR] {e}")
        return 0

def save_model_info(accuracy, method):
    """Store model training results for the UI (and change delta)."""
    try:
        _ensure_model_info_schema()
        # store as 0–100 percentage
        if accuracy is not None:
            try:
                a = float(accuracy)
                accuracy = (a * 100.0) if a <= 1.001 else a
            except Exception:
                pass
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO model_info (accuracy, method, updated_at) VALUES (%s, %s, %s)",
                (accuracy, method, datetime.now(timezone.utc))
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[MODEL INFO SAVE ERROR] {e}")
    
def get_model_metadata():
    """
    Unified model metadata for Control page.
    - Primary: 'model_info' table (accuracy + change)
    - Sidecar fallback/merge: model_metadata.json (updated_at, source)
    - Enrich: latest row from 'model_metadata' (version, features, labels)
    """
    out = {"accuracy": None, "updated_at": None, "method": None, "change": None,
           "version": None, "features": [], "labels": []}
    # 1) UI table (accuracy + change)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM model_info ORDER BY updated_at DESC LIMIT 2")
            rows = cursor.fetchall() or []
            if rows:
                latest = rows[0]
                prev = rows[1] if len(rows) > 1 else None
                acc = latest['accuracy']
                out["accuracy"] = acc
                out["updated_at"] = latest['updated_at'].strftime("%Y-%m-%d %H:%M:%S")
                out["method"] = latest['method']
                if prev and acc is not None and prev['accuracy'] is not None:
                    out["change"] = round(float(acc) - float(prev['accuracy']), 2)
    except Exception as e:
        logger.error(f"[MODEL INFO LOAD ERROR] {e}")
    # 2) Sidecar merge (fallback for updated_at/accuracy when UI table absent)
    try:
        if os.path.exists("model_metadata.json"):
            with open("model_metadata.json", "r") as f:
                sc = json.load(f)
            if out["accuracy"] is None and sc.get("accuracy") is not None:
                out["accuracy"] = sc.get("accuracy")
            out["updated_at"] = out["updated_at"] or sc.get("updated_at")
            out["method"] = out["method"] or sc.get("source")
    except Exception as e:
        logger.warning(f"[MODEL sidecar read] {e}")
    # 3) Enrich from authoritative metadata
    try:
        row = get_latest_model_metadata_db()
        if row:
            # DictRow or tuple – access defensively
            version = row["version"] if isinstance(row, dict) else row[1]
            feats   = row["features"] if isinstance(row, dict) else row[2]
            labels  = row["labels"] if isinstance(row, dict) else row[4]
            out["version"]  = version
            out["features"] = feats or []
            out["labels"]   = labels or []
    except Exception as e:
        logger.warning(f"[MODEL METADATA ENRICH] {e}")

    # --- FINAL NORMALIZATION (ensure 0–100 for UI/API) ------------------------
    try:
        if out.get("accuracy") is not None:
            a = float(out["accuracy"])
            out["accuracy"] = (a * 100.0) if a <= 1.001 else a
    except Exception:
        # leave as-is if anything odd comes through
        pass
    return out

# ---- MJPEG helpers ----
def _shared_static(filename: str) -> str:
    """Resolve a shared static directory (writer == reader)."""
    base = os.environ.get("ISK_STATIC_DIR", os.path.join(os.path.dirname(__file__), "static"))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)

def _resolve_heatmap_path() -> str:
    """
    Prefer a full composite 'heatmap_3d.png' if present; otherwise fall back to
    any of the plane projections the plotter leaves: xy/xz/yz.
    """
    candidates = ("heatmap_3d.png", "heatmap_xy.png", "heatmap_xz.png", "heatmap_yz.png")
    for name in candidates:
        p = _shared_static(name)
        try:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        except Exception:
            pass
            # last resort: point at the nominal 3d name (streamer will show fallback frame until it appears)
    return _shared_static("heatmap_3d.png")

def _mjpeg_stream_from(path, fps=2):
    """Stream a single image file (PNG/JPEG) as MJPEG, with retry + caching."""
    interval = 1.0 / max(fps, 1)
    last_mtime = None
    last_frame = b""

    def _frame_bytes():
        nonlocal last_mtime, last_frame
        # Try a few times to ride out atomic swaps & FS lag
        for _ in range(3):
            try:
                mtime = os.path.getmtime(path)
                # If unchanged and we have a cached frame, reuse it
                if last_mtime == mtime and last_frame:
                    return last_frame
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None or img.size == 0:
                    time.sleep(0.05)
                    continue
                # Normalize channels so JPEG encode always works
                if img.ndim == 3 and img.shape[2] == 4:      # BGRA -> BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.ndim == 2:                          # Gray -> BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                ok, buf = cv2.imencode(".jpg", img)
                if ok:
                    last_mtime = mtime
                    last_frame = bytes(buf)
                    return last_frame
            except Exception:
                time.sleep(0.05)
        # Fallback: small gray frame so the stream keeps flowing
        try:
            fallback = (np.ones((240, 320, 3), dtype=np.uint8) * 24)
            img = (np.ones((180, 240, 3), dtype=np.uint8) * 16)
            cv2.putText(img, "Waiting for radar...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img)
            return bytes(buf) if ok else (last_frame or b"")
        except Exception:
            pass
        return b""

    def _gen():
        while True:
            frame = _frame_bytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(interval)

    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.before_request
def _touch_user_activity():
    try:
        if not current_user.is_authenticated:
            return

        ua = (request.headers.get("User-Agent") or "")[:512]
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Preferred path (if the extra columns exist)
                    cur.execute("""
                        INSERT INTO user_activity (user_id, last_activity, last_ip, user_agent)
                        VALUES (%s, NOW(), %s, %s)
                        ON CONFLICT (user_id) DO UPDATE
                        SET last_activity = EXCLUDED.last_activity,
                            last_ip       = EXCLUDED.last_ip,
                            user_agent    = EXCLUDED.user_agent
                    """, (current_user.id, ip, ua))
                except psycopg2.errors.UndefinedColumn:
                    # Older schema → fall back to minimal upsert
                    conn.rollback()
                    cur.execute("""
                        INSERT INTO user_activity (user_id, last_activity)
                        VALUES (%s, NOW())
                        ON CONFLICT (user_id) DO UPDATE
                        SET last_activity = EXCLUDED.last_activity
                    """, (current_user.id,))
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_activity (
                            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                            last_activity TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    conn.commit()
                    cur.execute("""
                        INSERT INTO user_activity (user_id, last_activity)
                        VALUES (%s, NOW())
                        ON CONFLICT (user_id) DO UPDATE
                        SET last_activity = EXCLUDED.last_activity
                    """, (current_user.id,))
            conn.commit()
    except Exception as e:
        logger.debug(f"[USER_ACTIVITY] {e}")
    
def create_app():
    """Application factory pattern"""
    global app
    app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
    app.permanent_session_lifetime = timedelta(minutes=30)

    global radar
    try:
        rip = (config.get("radar_over_ip") or {})
        is_pc_consumer = bool(rip.get("enabled")) and str(rip.get("role", "pc")).lower() == "pc"
    except Exception:
        is_pc_consumer = False
    if is_pc_consumer:
        radar = None  
    else:
        try:
            radar = IWR6843Interface()
        except Exception as e:
            logger.warning(f"[RADAR] Not initializing serial interface here: {e}")
            radar = None

    def load_classifier():
        return ObjectClassifier()

    classifier = load_classifier()
    tracker = ObjectTracker(speed_limits_map=config.get("dynamic_speed_limits", {}))

    # Flask-Login
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)

    @app.context_processor
    def inject_globals():
        # expose callable so templates can use {{ is_admin() }}
        return {"now": datetime.now(IST), "is_admin": is_admin}

    @app.context_processor
    def inject_ptz_helpers():
        try:
            return {"ptz_capable": _ptz_capable()}
        except Exception:
            return {"ptz_capable": False}
        
    # Prefer JSON for AJAX/API calls so the frontend never tries to parse HTML as JSON
    def _wants_json():
        accept = (request.headers.get("Accept") or "").lower()
        ajax   = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        json_ep = request.path.startswith("/api/") or request.path in {
            "/retrain_model", "/upload_model", "/toggle_camera",
            "/manual_snapshot", "/mark_snapshot"
        }
        return ("application/json" in accept) or ajax or json_ep

    @app.errorhandler(404)
    def not_found(error):
        if _wants_json():
            return jsonify({"error": "Not found"}), 404
        return render_template('errors.html', message="Page not found", error_code=404), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors.html', message="Internal server error", error_code=500), 500
    
    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(
            os.path.join(app.root_path, 'static'),
            'favicon.ico',
            mimetype='image/vnd.microsoft.icon'
        )

    @app.errorhandler(413)
    def file_too_large(error):
        if _wants_json():
            return jsonify({"error": "File too large"}), 413
        return render_template('errors.html', message="File too large", error_code=413), 413

    @app.errorhandler(405)
    def method_not_allowed(error):
        # Prefer JSON for API callers so 405 isn't wrapped as a 500.
        if _wants_json():
            allowed = list(getattr(error, "valid_methods", []) or [])
            return jsonify({"error": "Method Not Allowed", "allow": allowed}), 405
        return render_template('errors.html', message="Method Not Allowed", error_code=405), 405

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        logger.exception("[ERROR] Unhandled exception")
        if _wants_json():
            return jsonify({"error": "Internal server error", "details": str(error)}), 500
        return render_template('errors.html', message="Internal error", error_code=500), 500
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Live MJPEG streams for Radar Visualizer panes
    # - Streams latest plot images that the plotter writes:
    #   * scatter_3d.png (3D point cloud)
    #   * tm2d.png       (Top-down TM panel)
    #   Written under ISK_STATIC_DIR or ./static by Live3DPlotter.  :contentReference[oaicite:2]{index=2}
    # ──────────────────────────────────────────────────────────────────────────────
    def _resolve_static_dir():
        """Match the directory where the plotter saves frames."""
        return os.environ.get(
            "ISK_STATIC_DIR",
            os.path.join(os.path.dirname(__file__), "static")
        )

    def _mjpeg_stream(image_path_resolver, fps=6):
        """
        Yield a multipart/x-mixed-replace (MJPEG) stream of the latest image.
        Re-encodes PNG→JPEG and never stalls: emits a fallback frame if missing.
        """
        frame_interval = max(0.001, 1.0 / float(fps))
        last_mtime = -1.0
        fallback = (np.ones((240, 320, 3), dtype=np.uint8) * 24)
        cv2.putText(fallback, "Waiting for radar…", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        fallback_jpg = cv2.imencode(".jpg", fallback)[1].tobytes()
        # cache last good frame to avoid blips when source file timestamp doesn't change
        last_good_jpg = fallback_jpg

        while True:
            try:
                path = image_path_resolver()
                jpg = None

                if path and os.path.isfile(path):
                    mtime = os.path.getmtime(path)
                    # send on first seen or when updated
                    if mtime != last_mtime:
                        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        if img is None or img.size == 0:
                            jpg = fallback_jpg
                        else:
                            # Normalize channels: BGRA/GRAY → BGR
                            if img.ndim == 3 and img.shape[2] == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                            elif img.ndim == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            jpg = buf.tobytes() if ok else last_good_jpg
                            if ok:
                                last_good_jpg = jpg
                        last_mtime = mtime
                    else:
                        # unchanged: reuse fallback to keep stream ticking
                        jpg = last_good_jpg
                else:
                    # file doesn’t exist yet
                    jpg = last_good_jpg

                yield (b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                    jpg + b"\r\n")
                time.sleep(frame_interval)
            except GeneratorExit:
                break
            except Exception:
                # keep alive on transient errors
                time.sleep(0.5)

    @app.route("/viz/pc.mjpg")
    def viz_pc_mjpg():
        """Continuous 3D point-cloud stream from scatter_3d.png."""
        def _pc_path():
            return os.path.join(_resolve_static_dir(), "scatter_3d.png")
        return Response(_mjpeg_stream(_pc_path, fps=12),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/viz/tm2d.mjpg")
    def viz_tm2d_mjpg():
        """Continuous Top-down TM stream from tm2d.png."""
        def _tm2d_path():
            return os.path.join(_resolve_static_dir(), "tm2d.png")
        resp = Response(_mjpeg_stream(_tm2d_path, fps=12),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.route("/viz/pc_full.mjpg")
    def viz_pc_full_mjpg():
        """Continuous 3D point-cloud (full-view) from scatter_3d_full.png."""
        def _pc_full_path():
            return os.path.join(_resolve_static_dir(), "scatter_3d_full.png")
        resp = Response(_mjpeg_stream(_pc_full_path, fps=12),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.route("/viz/tm2d_full.mjpg")
    def viz_tm2d_full_mjpg():
        """Continuous Top-down TM (full-view) from tm2d_full.png."""
        def _tm2d_full_path():
            return os.path.join(_resolve_static_dir(), "tm2d_full.png")
        resp = Response(_mjpeg_stream(_tm2d_full_path, fps=12),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # ───────────────────────── Pi Health ─────────────────────────
    def _fetch_pi_health(timeout_s: float = 1.5) -> dict:
        """
        Best-effort fetch of Pi /health.
        Returns {ok, temperature, cpu_load, radar_connected} with None defaults on failure.
        """
        url = PI_HEALTH_URL
        if not url:
            return {"ok": False, "temperature": None, "cpu_load": None, "radar_connected": None}
        try:
            r = requests.get(url, timeout=timeout_s)
            if r.ok:
                j = r.json() or {}
                return {
                    "ok": True,
                    "temperature": j.get("temperature"),
                    "cpu_load": j.get("cpu_load"),
                    "radar_connected": j.get("radar_connected")
                }
        except Exception:
            pass
        return {"ok": False, "temperature": None, "cpu_load": None, "radar_connected": None}

    @app.get("/api/pi_health")
    def api_pi_health():
        """Proxy Pi’s /health so the frontend can fetch from same origin."""
        h = _fetch_pi_health()
        code = 200 if h.get("ok") else 502
        return jsonify(h), code

    # ───────────────────────── Map (Leaflet) ─────────────────────────
    @app.route("/api/location")
    @login_required
    def api_location():
        """
        Current location for the map.
        Prefers GPS (via gpsd) when enabled in config, else static site coords.
        """
        try:
            loc = get_current_location(config)
            return jsonify({
                "ok": True,
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
                "alt": loc.get("alt"),
                "hdop": loc.get("hdop"),
                "time": loc.get("time"),
                "source": loc.get("source"),
                "site_id": loc.get("site_id"),
                "name": loc.get("name"),
                "address": loc.get("address"),
            })
        except Exception as e:
            logger.error(f"[MAP] api_location failed: {e}")
            return jsonify({"ok": False, "error": "location_unavailable"}), 503
        
    @app.route("/api/map_sensors")
    @login_required
    def api_map_sensors():
        """
        Return cameras (from DB if available, else config) and radars (from config)
        with positions + FOV/bearing/range for drawing coverage sectors.
        Shape:
        {
            ok: True,
            site: {lat, lon, name, address},
            sensors: [
            {type:"camera"| "radar", name, lat, lon, bearing, fov, range, color}
            ]
        }
        """
        try:
            site = get_current_location(config)  # already GPS-aware fallback
            site_lat = float(site.get("lat") or 0.0)
            site_lon = float(site.get("lon") or 0.0)

            # Prefer DB cameras (multiple + roles), fallback to file config
            try:
                cams = _active_cameras_from_db()
            except Exception:
                cams = (config.get("cameras") or [])
                if isinstance(cams, dict):
                    cams = [cams]

            camera_default_fov = float(config.get("camera_fov_h_deg", 90.0))

            palette = ["#2563EB", "#38BDF8", "#A78BFA", "#0EA5E9", "#34D399",
                    "#F59E0B", "#EC4899", "#22C55E"]

            sensors = []

            # Cameras
            for i, c in enumerate(cams or []):
                if not c.get("enabled", True):
                    continue
                lat = float(c.get("lat") or site_lat)
                lon = float(c.get("lon") or site_lon)
                fov = float(c.get("fov_deg") or camera_default_fov)
                bearing = float(c.get("bearing_deg") or c.get("yaw_deg") or 0.0)
                rng = float(c.get("range_m") or c.get("range") or 120.0)
                sensors.append({
                    "type": "camera",
                    "name": c.get("name") or f"Camera {i+1}",
                    "lat": lat, "lon": lon,
                    "bearing": bearing, "fov": fov, "range": rng,
                    "color": palette[i % len(palette)]
                })

            # Radars
            radars = config.get("radars")
            if not radars:
                # Back-compat single radar config
                r = (config.get("radar") or {})
                if r:
                    radars = [r]
            for j, r in enumerate(radars or []):
                lat = float(r.get("lat") or site_lat)
                lon = float(r.get("lon") or site_lon)
                fov = float(r.get("fov_deg") or r.get("fov") or 120.0)
                bearing = float(r.get("bearing_deg") or r.get("bearing") or 0.0)
                rng = float(r.get("max_range_m") or r.get("range") or 150.0)
                sensors.append({
                    "type": "radar",
                    "name": r.get("name") or f"Radar {j+1}",
                    "lat": lat, "lon": lon,
                    "bearing": bearing, "fov": fov, "range": rng,
                    "color": "#10B981"  # steady green for radar
                })

            return jsonify({
                "ok": True,
                "site": {
                    "lat": site_lat, "lon": site_lon,
                    "name": site.get("name") or "Site",
                    "address": site.get("address") or ""
                },
                "sensors": sensors
            })
        except Exception as e:
            logger.error(f"[MAP] api_map_sensors failed: {e}")
            return jsonify({"ok": False}), 500

    @app.route("/map")
    @login_required
    def map_page():
        """
        Full-screen Leaflet map that centers on the configured (or GPS) location.
        """
        # Provide a first paint using whatever we have synchronously,
        # the page will then refresh via /api/location.
        loc = get_current_location(config)
        return render_template(
            "map.html",
            initial_lat=loc.get("lat"),
            initial_lon=loc.get("lon"),
            initial_name=loc.get("name") or "Site",
            initial_addr=loc.get("address") or "",
            initial_source=loc.get("source") or "static",
            initial_hdop=loc.get("hdop"),
            initial_time=loc.get("time"),
        )

    @app.route("/camera_feed")
    @login_required
    def camera_feed():
        def generate_heatmap_frame():
            try:
                with open(_resolve_heatmap_path(), "rb") as f:
                    return f.read()
            except Exception:
                img = (np.ones((240, 320, 3), dtype=np.uint8) * 16)
                _, buf = cv2.imencode(".jpg", img)
                return bytes(buf)

        # Stream a sequence of HTTP snapshots as MJPEG
        def _stream_snapshot_loop(url, auth, interval=0.5):
            while True:
                try:
                    r = requests.get(url, auth=auth, timeout=5)
                    if r.status_code == 200 and r.content.startswith(b'\xff\xd8'):
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + r.content + b"\r\n")
                    else:
                        logger.warning(f"[CAMERA_FEED] Snapshot failed, status={r.status_code}")
                except Exception as e:
                    logger.error(f"[CAMERA_FEED] Snapshot error: {e}")
                time.sleep(interval)

        def generate():
            try:
                # Prefer DB; if it hiccups, fall back to file config so the feed never dies
                try:
                    config["cameras"], config["selected_camera"] = load_cameras_from_db()
                except Exception as e:
                    logger.warning(f"[CAMERA_FEED] DB load failed; using file config: {e}")
                    cfg_file = load_config()
                    config["cameras"] = cfg_file.get("cameras", [])
                    config["selected_camera"] = cfg_file.get("selected_camera", 0)

                cam = config["cameras"][config["selected_camera"]] if config["cameras"] else {}

                camera_enabled = cam.get("enabled", True)
                stream_type = str(cam.get("stream_type", "snapshot")).lower()
                username = cam.get("username")
                password = cam.get("password")
                auth = HTTPDigestAuth(username, password) if username and password else None

                # Determine stream URL(s)
                url = cam.get("url")
                snapshot_url = cam.get("snapshot_url") or url
                if not url:
                    ip = cam.get("ip")
                    if stream_type == "snapshot":
                        url = f"http://{ip}/axis-cgi/jpg/image.cgi"
                    elif stream_type == "mjpeg":
                        url = f"http://{ip}/mjpg/video.mjpg"
                    elif stream_type == "rtsp":
                        url = f"rtsp://{ip}/axis-media/media.amp"
                    if not snapshot_url:
                        snapshot_url = f"http://{ip}/axis-cgi/jpg/image.cgi" if ip else None

                if not camera_enabled or not url:
                    logger.warning("[CAMERA_FEED] Falling back to 3D heatmap (camera disabled or missing)")
                    def heatmap_stream():
                        while True:
                            frame = generate_heatmap_frame()
                            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                            time.sleep(0.5)
                    return Response(heatmap_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

                if stream_type == "rtsp":
                    # RTSP→MJPEG via FFmpeg; auto-fallback to snapshots if missing/flaky
                    ff = _ffmpeg_bin()
                    if not ff:
                        logger.error("[CAMERA_FEED] ffmpeg not found; falling back to snapshot loop")
                        return Response(_stream_snapshot_loop(snapshot_url, auth), mimetype='multipart/x-mixed-replace; boundary=frame')

                    rtsp_url = _embed_basic_in_rtsp(url, username, password)

                    ffmpeg_cmd = [
                        ff, "-rtsp_transport", "tcp",
                        "-user_agent", "Mozilla/5.0",
                        "-i", rtsp_url,
                        "-f", "mjpeg", "-qscale:v", "2", "-r", "5", "-"
                    ]
                    while True:
                        process = subprocess.Popen(
                            ffmpeg_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            startupinfo=_si_no_window(),
                            creationflags=_cf_no_window()
                        )
                        buffer = b""
                        got_frame = False
                        start_ts = time.time()
                        try:
                            while True:
                                chunk = process.stdout.read(4096)
                                if not chunk:
                                    if (time.time() - start_ts) > 3.0 and not got_frame:
                                        logger.warning("[CAMERA_FEED] No RTSP frames yet; retrying…")
                                        break
                                    time.sleep(0.05)
                                    continue
                                buffer += chunk
                                s = buffer.find(b'\xff\xd8')
                                e = buffer.find(b'\xff\xd9')
                                if s != -1 and e != -1 and e > s:
                                    frame = buffer[s:e+2]
                                    buffer = buffer[e+2:]
                                    got_frame = True
                                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        finally:
                            try:
                                process.kill()
                            except Exception:
                                pass

                        if not got_frame:
                            logger.warning("[CAMERA_FEED] Falling back to snapshot loop for 10s")
                            fallback_until = time.time() + 10
                            for part in _stream_snapshot_loop(snapshot_url, auth, interval=0.6):
                                yield part
                                if time.time() > fallback_until:
                                    break

                elif stream_type == "snapshot":
                    # Prefer explicit snapshot_url if configured
                    yield from _stream_snapshot_loop(snapshot_url, auth, interval=0.5)

                else:
                    # MJPEG stream fallback
                    try:
                        with requests.get(url, auth=auth, stream=True, timeout=10) as r:
                            if r.status_code != 200:
                                logger.error(f"[CAMERA_FEED] MJPEG stream returned {r.status_code}")
                                return
                            buffer = b""
                            for chunk in r.iter_content(chunk_size=4096):
                                buffer += chunk
                                start = buffer.find(b'\xff\xd8')
                                end = buffer.find(b'\xff\xd9')
                                if start != -1 and end != -1 and end > start:
                                    frame = buffer[start:end+2]
                                    buffer = buffer[end+2:]
                                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    except Exception as e:
                        logger.error(f"[CAMERA_FEED] MJPEG stream error: {e}")

            except Exception as e:
                logger.exception(f"[CAMERA_FEED] Fatal error: {e}")
                time.sleep(2)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # --- MP4 streaming with HTTP Range (Safari/Chrome friendly) -------------
    def _iter_file_range(path, start, end, chunk_size=1024 * 1024):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)

    @app.route("/clips/<path:filename>", methods=["GET", "HEAD"])
    @login_required
    def serve_clip(filename):
        # Always resolve inside CLIPS_FOLDER and strip any path tricks
        safe_name = os.path.basename(filename)
        abs_path = os.path.join(CLIPS_FOLDER, safe_name)
        if not os.path.exists(abs_path):
            return "Clip not found", 404
        # Ensure the file is browser-friendly *before* any response
        try:
            _ensure_web_playable_mp4(abs_path)
        except Exception:
            pass
        file_size = os.path.getsize(abs_path)
        range_header = request.headers.get("Range", "").strip()
        # Strong caching hints for video scrubbing/autoplay
        mtime = os.path.getmtime(abs_path)
        last_modified = email.utils.formatdate(mtime, usegmt=True)
        etag = f'W/"{file_size:x}-{int(mtime):x}"'
        # HEAD preflight (some browsers do this before Range GET)
        if request.method == "HEAD":
            resp = Response(status=200)
            resp.headers["Content-Type"] = "video/mp4"
            resp.headers["Content-Length"] = str(file_size)
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Disposition"] = f'inline; filename="{safe_name}"'
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
            resp.headers["Last-Modified"] = last_modified
            resp.headers["ETag"] = etag
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["X-Accel-Buffering"] = "no"
            return resp
        if range_header.startswith("bytes="):
            try:
                # e.g. "bytes=12345-" or "bytes=0-1023"
                start_s, end_s = range_header.split("=", 1)[1].split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else (file_size - 1)
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                length = end - start + 1
                resp = Response(
                    _iter_file_range(abs_path, start, end),
                    status=206,
                    mimetype="video/mp4",
                    direct_passthrough=True,
                )
                resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                resp.headers["Accept-Ranges"] = "bytes"
                resp.headers["Content-Length"] = str(length)
                resp.headers["Content-Disposition"] = f'inline; filename="{safe_name}"'
                # help some proxies/browsers stream smoothly
                resp.headers["Cache-Control"] = "no-store, must-revalidate"
                resp.headers["Last-Modified"] = last_modified
                resp.headers["ETag"] = etag
                resp.headers["X-Content-Type-Options"] = "nosniff"
                resp.headers["X-Accel-Buffering"] = "no"
                return resp
            except Exception:
                # fall back to full file on any parse hiccup
                pass

        # No Range requested: serve whole file (most browsers still switch to Range after)
        resp = send_from_directory(CLIPS_FOLDER, safe_name, mimetype="video/mp4", as_attachment=False)
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Disposition"] = f'inline; filename="{safe_name}"'
        resp.headers["Cache-Control"] = "private, max-age=3600, must-revalidate"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/api/clip_status")
    @login_required
    def api_clip_status():
        """
        Query by snapshot filename; returns clip readiness and URL if available.
        Example: /api/clip_status?snapshot=annotated_20250822_101530.jpg
        """
        snap = (request.args.get("snapshot") or "").strip()
        if not snap:
            return jsonify({"error": "snapshot required"}), 400

        with get_db_connection() as conn:
            row = _row_by_snapshot_basename(conn, snap)
        if not row:
            return jsonify({"error": "not found"}), 404

        ready = _is_clip_ready(row)
        payload = {
            "status": (row.get("clip_status") or "").lower() if row.get("clip_status") else None,
            "ready": bool(ready),
            "duration_s": float(row["clip_duration_s"]) if row.get("clip_duration_s") is not None else None,
            "fps": float(row["clip_fps"]) if row.get("clip_fps") is not None else None,
            "size_bytes": int(row["clip_size_bytes"]) if row.get("clip_size_bytes") is not None else None,
            "gdrive_link": row.get("clip_gdrive_link"),
            "clip_url": None,
            "clip_filename": os.path.basename(row["clip_path"]) if row.get("clip_path") else None
        }
        if ready:
            payload["clip_url"] = url_for("serve_clip", filename=os.path.basename(row["clip_path"]))
            return jsonify(payload), 200

        # Fallback to violation bundle's clip.mp4
        bdir = _resolve_bundle_by_basename(snap)
        if bdir:
            bp = os.path.join(bdir, "clip.mp4")
            if os.path.isfile(bp):
                payload.update({
                    "status": "ready",
                    "ready": True,
                    "clip_url": url_for("bundle_asset", snapshot=snap, name="clip.mp4"),
                    "clip_filename": "clip.mp4",
                })
                return jsonify(payload), 200

        # Still pending/missing
        return jsonify(payload), 202
    
    @app.route("/api/clip_upload", methods=["POST"])
    @login_required
    def api_clip_upload():
        """
        Force (re)upload of a ready clip to Google Drive for a given snapshot basename.
        Body (JSON): { "snapshot": "<basename.jpg>" }
        Updates radar_data.clip_gdrive_link on success.
        """
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            data = request.get_json(silent=True) or {}
            snap = (data.get("snapshot") or "").strip()
            if not snap:
                return jsonify({"error": "snapshot required"}), 400
            with get_db_connection() as conn:
                row = _row_by_snapshot_basename(conn, snap)
                if not row:
                    return jsonify({"error": "not found"}), 404
                if not _is_clip_ready(row):
                    return jsonify({"error": "clip not ready"}), 409
                local_fp = (row.get("clip_path") or "").strip()
                if not (local_fp and os.path.exists(local_fp)):
                    return jsonify({"error": "local clip file missing"}), 410
                try:
                    _ensure_web_playable_mp4(local_fp)
                except Exception:
                    pass
                url = _rclone_copy_and_link(local_fp)
                if url:
                    _ = _update_clip_fields_by_basename(conn, snap, clip_gdrive_link=url)
                return jsonify({"status": "ok", "gdrive_link": url})
        except Exception as e:
            logger.exception("[CLIP UPLOAD] error")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    @app.route("/bundle/asset", methods=["GET","HEAD"])
    @login_required
    def bundle_asset():
        """
        Serve a file from a violation bundle by snapshot basename and filename.
        GET/HEAD ?snapshot=<basename.jpg>&name=<file-in-bundle>
        For video, supports HEAD + Range for inline playback/scrubbing.
        """
        snap = (request.args.get("snapshot") or "").strip()
        name = (request.args.get("name") or "").strip()
        if not snap or not name:
            return "snapshot and name required", 400

        bdir = _resolve_bundle_by_basename(snap)
        if not bdir or not os.path.isdir(bdir):
            return "bundle not found", 404

        # normalize and block traversal
        rel = os.path.normpath(name).replace("\\", "/")
        if rel.startswith("../") or rel.startswith("..\\") or rel.startswith("/"):
            return "invalid path", 400
        abs_path = os.path.join(bdir, rel)
        if not os.path.realpath(abs_path).startswith(os.path.realpath(bdir) + os.sep):
            return "invalid path", 400
        if not os.path.isfile(abs_path):
            return "file not found", 404

        ln = rel.lower()
        is_video = ln.endswith((".mp4", ".mov", ".m4v"))

        if not is_video:
            # images / other assets
            mt = (
                "image/jpeg" if ln.endswith((".jpg",".jpeg")) else
                "image/png" if ln.endswith(".png") else
                "application/octet-stream"
            )
            return send_file(abs_path, mimetype=mt, as_attachment=False)

        # Videos: ensure streamable, then serve with Range support
        serve_path, mime = _ensure_streamable_video(abs_path)
        try:
            file_size = os.path.getsize(serve_path)
        except Exception:
            file_size = os.path.getsize(abs_path)
            serve_path = abs_path
            mime = "video/mp4" if ln.endswith(".mp4") else "video/quicktime"

        range_header = (request.headers.get("Range") or "").strip()
        try:
            mtime = os.path.getmtime(serve_path)
        except Exception:
            mtime = os.path.getmtime(abs_path)
        last_modified = email.utils.formatdate(mtime, usegmt=True)

        # HEAD preflight
        if request.method == "HEAD":
            resp = Response(status=200, mimetype=mime)
            resp.headers["Content-Length"] = str(file_size)
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Last-Modified"] = last_modified
            resp.headers["Content-Disposition"] = f'inline; filename="{os.path.basename(serve_path)}"'
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["X-Accel-Buffering"] = "no"
            return resp

        # Byte-range
        if range_header.startswith("bytes="):
            try:
                start_s, end_s = range_header.split("=", 1)[1].split("-", 1)
                start = int(start_s) if start_s else 0
                end   = int(end_s) if end_s else (file_size - 1)
                start = max(0, min(start, file_size - 1))
                end   = max(start, min(end, file_size - 1))
                length = end - start + 1
                resp = Response(
                    _iter_file_range(serve_path, start, end),
                    status=206, mimetype=mime, direct_passthrough=True
                )
                resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                resp.headers["Accept-Ranges"] = "bytes"
                resp.headers["Content-Length"] = str(length)
                resp.headers["Content-Disposition"] = f'inline; filename="{os.path.basename(serve_path)}"'
                resp.headers["Cache-Control"] = "private, max-age=3600, must-revalidate"
                resp.headers["X-Content-Type-Options"] = "nosniff"
                resp.headers["X-Accel-Buffering"] = "no"
                return resp
            except Exception:
                pass

        # Full-file fallback (moov is at head, so progressive works)
        resp = send_file(serve_path, mimetype=mime, as_attachment=False)
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Disposition"] = f'inline; filename="{os.path.basename(serve_path)}"'
        resp.headers["Cache-Control"] = "private, max-age=3600, must-revalidate"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp


    @app.route("/api/bundle_manifest")
    @login_required
    def api_bundle_manifest():
        """
        JSON manifest for a violation bundle, keyed by snapshot basename.
        ?snapshot=<basename.jpg>
        Returns images[] and videos[] with ready-to-use URLs.
        """
        snap = (request.args.get("snapshot") or "").strip()
        if not snap:
            return jsonify({"error": "snapshot required"}), 400

        bdir = _resolve_bundle_by_basename(snap)
        if not bdir or not os.path.isdir(bdir):
            return jsonify({"error": "bundle not found"}), 404

        items = sorted(os.listdir(bdir))
        images, videos = [], []

        for name in items:
            ln = name.lower()
            if ln.endswith((".jpg", ".jpeg", ".png")):
                images.append({
                    "name": name,
                    "url": url_for("bundle_asset", snapshot=snap, name=name, _external=False)
                })
            if ln.endswith((".mp4", ".mov", ".m4v")):
                videos.append({
                    "name": name,
                    "play_url": url_for("bundle_asset", snapshot=snap, name=name, _external=False)
                })

        return jsonify({
            "snapshot": snap,
            "bundle_dir": bdir,
            "zip_url": url_for("bundle_zip", snapshot=snap, _external=False),
            "images": images,
            "videos": videos
        })

    @app.route("/api/cameras/toggle", methods=["POST"])
    @login_required
    def api_camera_toggle():
        """
        Toggle a specific camera's active state (multi-camera friendly).
        Body can be form or JSON with fields:
          - cam_id (int, required)
          - active (optional bool). If omitted, the state will be toggled.
        """
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            data = request.get_json(silent=True) or request.form or {}
            cam_id = int(data.get("cam_id"))
            # Read current state from DB
            cams, _ = load_cameras_from_db()
            cam = next((c for c in cams if int(c.get("id", -1)) == cam_id), None)
            if not cam:
                return jsonify({"error": "Camera not found"}), 404
            desired = data.get("active")
            if desired is None:
                # toggle if no explicit state given
                desired = (not cam.get("enabled", True))
            desired = True if str(desired).lower() in ("1","true","on","yes") else False
            set_camera_active_state(cam_id, desired)
            return jsonify({"status": "ok", "cam_id": cam_id, "active": desired})
        except Exception as e:
            logger.error(f"[CAMERA TOGGLE ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500                                                                                                               

    @app.route("/api/cameras", methods=["GET"])
    @login_required
    def api_list_cameras():
        """
        Return all cameras with their active state so the UI can allow multiple actives.
        """
        cams, sel = load_cameras_from_db()
        # Do not leak credentials
        safe = []
        for c in cams:
            safe.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "stream_type": c.get("stream_type"),
                "enabled": bool(c.get("enabled", True)),
            })
        return jsonify({"cameras": safe, "selected_idx": sel})

    @app.route("/api/ptz/mode", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_mode():
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        body = request.get_json(silent=True) or {}
        st = _read_ptz_state()
        # Accept JSON or query args; default to toggle when unspecified
        enabled = body.get("enabled", None)
        if enabled is None:
            q = request.args.get("enabled")
            if q is not None:
                enabled = str(q).lower() in ("1","true","yes","on")
        if enabled is None and str(request.args.get("toggle","")).lower() in ("1","true","yes","on"):
            enabled = not bool(st.get("enabled", False))
        if enabled is None:
            enabled = not bool(st.get("enabled", False))
        st["enabled"] = bool(enabled)
        ok = _write_ptz_state(st)
        # mirror status (best-effort)
        try:
            tmp = PTZ_STATUS_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"enabled": bool(st.get("enabled")), "locked_by": ("auto" if st.get("enabled") else ""), "updated_at": time.time()}, f)
            os.replace(tmp, PTZ_STATUS_PATH)
        except Exception:
            pass
        return jsonify({"ok": ok, "state": st}), (200 if ok else 500)

    @app.route("/api/ptz/lock", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_lock():
        # CORS preflight
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        # Accept both JSON and query params; support multiple aliases for the track id.
        body = request.get_json(silent=True) or {}
        # gather 'clear' from body or query, normalize to bool
        clear = body.get("clear", None)
        if clear is None:
            q_clear = request.args.get("clear")
            if q_clear is not None:
                clear = str(q_clear).lower() in ("1","true","yes","on")
        clear = bool(clear) if clear is not None else False

        # collect tid from any alias in body or query
        tid = (
            body.get("tid")
            or body.get("id")
            or body.get("track_id")         
        )
        if tid is None:
            tid = (
                request.args.get("tid")
                or request.args.get("id")
                or request.args.get("track_id")  
            )
        tid = (str(tid).strip() if tid is not None else None)

        st = _read_ptz_state()
        prev = st.get("lock_tid")

        if clear:
            st["lock_tid"] = None
        elif tid:
            st["lock_tid"] = tid
        else:
            # No tid provided and not a clear request → preserve existing lock.
            st["lock_tid"] = prev

        ok = _write_ptz_state(st)
        try:
            logger.info(f"[PTZ] lock_tid set to {st['lock_tid']!r} (clear={clear})")
        except Exception:
            pass
        return jsonify({"ok": ok, "state": st}), (200 if ok else 500)

    @app.get("/api/ptz/status")
    @login_required
    def api_ptz_status():
        try:
            stat = _read_ptz_status()
            st   = _read_ptz_state()
            return jsonify({
                "ok": True,
                "capable": _ptz_capable(),
                "status": {
                    "enabled": bool(stat.get("enabled", False)),
                    "locked_by": stat.get("locked_by") or "",
                    "updated_at": float(stat.get("updated_at") or 0.0),
                    "lock_tid": st.get("lock_tid") 
                }
            })
        except Exception:
            return jsonify({"ok": False, "error": "status_unavailable"}), 503

    @app.route("/api/ptz/nudge", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_nudge():
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        st = _read_ptz_status()
        if bool(st.get("enabled", False)):
            return jsonify({"ok": False, "error": "auto_track_active",
                            "message": "Auto-track is ON. Turn it OFF to control PTZ manually."}), 409
        body = request.get_json(silent=True) or {}
        # Accept both JSON and query parameters
        def _getf(key, default):
            if key in body: return body.get(key, default)
            v = request.args.get(key)
            return v if v is not None else default
        try:
            vx  = max(-1.0, min(1.0, float(_getf("vx", 0.0))))
            vy  = max(-1.0, min(1.0, float(_getf("vy", 0.0))))
            vz  = max(-1.0, min(1.0, float(_getf("vz", 0.0))))
            sec = max(0.05, min(2.0, float(_getf("sec", 0.35))))
        except Exception:
            return jsonify({"ok": False, "error": "invalid parameters"}), 400
        ptz_cfg = _resolve_ptz_cfg_from_env()
        ok = _ptz_direct_control({"type":"nudge","vx":vx,"vy":vy,"vz":vz,"sec":sec,"ptz_cfg": ptz_cfg})
        if ok:
            try:
                from main import _ptz_q as _MAIN_PTZ_Q
                _MAIN_PTZ_Q.put_nowait({"type": "nudge", "vx": vx, "vy": vy, "vz": vz, "sec": sec, "ptz_cfg": ptz_cfg})
            except Exception:
                pass
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "ptz_not_available"}), 500

    @app.route("/api/ptz/zoom", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_zoom():
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        st = _read_ptz_status()
        if bool(st.get("enabled", False)):
            return jsonify({"ok": False, "error": "auto_track_active",
                            "message": "Auto-track is ON. Turn it OFF to control PTZ manually."}), 409
        body = request.get_json(silent=True) or {}
        # Accept vz (float in [-1,1]) or dz in {-1,1}; and from query params too
        def _getf(key, default):
            if key in body: return body.get(key, default)
            v = request.args.get(key)
            return v if v is not None else default
        try:
            vz   = _getf("vz", None)
            dz   = _getf("dz", None)
            dir_ = _getf("dir", None)  # NEW: accept 'in'/'out'/+/- as alias
            if vz is None and dz is not None:
                vz = float(int(dz))  # -1 or +1
            if vz is None and dir_ is not None:
                s = str(dir_).strip().lower()
                if s in ("in", "+", "zoom_in"):   vz = +1.0
                elif s in ("out", "-", "zoom_out"): vz = -1.0
            vz  = max(-1.0, min(1.0, float(vz if vz is not None else 0.0)))
            sec = max(0.05, min(2.0, float(_getf("sec", 0.35))))
        except Exception:
            return jsonify({"ok": False, "error": "invalid parameters"}), 400
        ptz_cfg = _resolve_ptz_cfg_from_env()
        ok = _ptz_direct_control({"type":"zoom","vz":vz,"sec":sec,"ptz_cfg": ptz_cfg})
        if ok:
            try:
                from main import _ptz_q as _MAIN_PTZ_Q
                _MAIN_PTZ_Q.put_nowait({"type": "zoom", "vz": vz, "sec": sec, "ptz_cfg": ptz_cfg})
            except Exception:
                pass
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "ptz_not_available"}), 500

    @app.route("/api/ptz/stop", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_stop():
        """Hard stop — always allowed even if auto-track is active."""
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        ptz_cfg = _resolve_ptz_cfg_from_env()
        ok = _ptz_direct_control({"type":"stop","ptz_cfg": ptz_cfg})
        try:
            from main import _ptz_q as _MAIN_PTZ_Q
            _MAIN_PTZ_Q.put_nowait({"type": "stop", "ptz_cfg": ptz_cfg})
        except Exception:
            pass
        return jsonify({"ok": bool(ok)})

    @app.route("/api/ptz/move_abs", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_move_abs():
        """
        Absolute move. Body/Query accepts any subset of {pan, tilt, zoom} in camera units (degrees for pan/tilt).
        Guards manual control if auto-track is enabled.
        """
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        st = _read_ptz_status()
        if bool(st.get("enabled", False)):
            return jsonify({"ok": False, "error": "auto_track_active",
                            "message": "Auto-track is ON. Turn it OFF to control PTZ manually."}), 409
        data = request.get_json(silent=True) or {}
        def _getf(k):
            if k in data: return data.get(k)
            v = request.args.get(k)
            return v
        pan  = _getf("pan");  pan  = None if pan  is None else float(pan)
        tilt = _getf("tilt"); tilt = None if tilt is None else float(tilt)
        zoom = _getf("zoom"); zoom = None if zoom is None else float(zoom)
        ptz_cfg = _resolve_ptz_cfg_from_env()
        ok = _ptz_direct_control({"type":"abs","pan":pan,"tilt":tilt,"zoom":zoom,"ptz_cfg": ptz_cfg})
        if ok:
            try:
                from main import _ptz_q as _MAIN_PTZ_Q
                _MAIN_PTZ_Q.put_nowait({"type":"abs","pan":pan,"tilt":tilt,"zoom":zoom,"ptz_cfg": ptz_cfg})
            except Exception:
                pass
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "ptz_not_available"}), 500

    @app.route("/api/ptz/move_rel", methods=["POST","GET","OPTIONS"])
    @login_required
    def api_ptz_move_rel():
        """
        Relative move (step) in degrees/units. Accepts dpan, dtilt, dzoom.
        """
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        st = _read_ptz_status()
        if bool(st.get("enabled", False)):
            return jsonify({"ok": False, "error": "auto_track_active",
                            "message": "Auto-track is ON. Turn it OFF to control PTZ manually."}), 409
        data = request.get_json(silent=True) or {}
        def _getf(k, d=0.0):
            if k in data: return data.get(k, d)
            v = request.args.get(k)
            return v if v is not None else d
        try:
            dpan  = float(_getf("dpan", 0.0))
            dtilt = float(_getf("dtilt", 0.0))
            dzoom = float(_getf("dzoom", 0.0))
        except Exception:
            return jsonify({"ok": False, "error": "invalid parameters"}), 400
        ptz_cfg = _resolve_ptz_cfg_from_env()
        ok = _ptz_direct_control({"type":"rel","dpan":dpan,"dtilt":dtilt,"dzoom":dzoom,"ptz_cfg": ptz_cfg})
        if ok:
            try:
                from main import _ptz_q as _MAIN_PTZ_Q
                _MAIN_PTZ_Q.put_nowait({"type":"rel","dpan":dpan,"dtilt":dtilt,"dzoom":dzoom,"ptz_cfg": ptz_cfg})
            except Exception:
                pass
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "ptz_not_available"}), 500

    @app.get("/api/ptz/explain")
    @login_required
    def api_ptz_explain():
        """Return camera/PTZ capability text (from PTZClient.explain()), for UI hints."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        try:
            from ptz import PTZClient
        except Exception:
            try:
                from ptz_controller import PTZClient  # type: ignore
            except Exception:
                return jsonify({"ok": False, "error": "client_unavailable"}), 500
        cfg = _resolve_ptz_cfg_from_env()
        try:
            client = PTZClient(
                cfg.get("host"),
                cfg.get("username") or cfg.get("user") or "",
                cfg.get("password") or cfg.get("pwd") or "",
                port=int(cfg.get("port") or 80),
                auth_mode=(cfg.get("auth_mode") or cfg.get("auth") or "auto"),
            )
            text = client.explain()
            return jsonify({"ok": True, "explain": text})
        except Exception as e:
            return jsonify({"ok": False, "error": "unavailable", "details": str(e)}), 500

    @app.post("/api/ptz/preset")
    @login_required
    def api_ptz_preset():
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        body = request.get_json(silent=True) or {}
        preset = (str(body.get("preset") or "")).strip()
        if not preset:
            return jsonify({"ok": False, "error": "preset_required"}), 400
        try:
            from main import _ptz_q as _MAIN_PTZ_Q
            _MAIN_PTZ_Q.put_nowait({"type": "preset", "preset": preset, "ptz_cfg": _resolve_ptz_cfg_from_env()})
            return jsonify({"ok": True})
        except Exception:
            ok = _ptz_direct_control({"type":"preset","preset":preset,"ptz_cfg": _resolve_ptz_cfg_from_env()})
            return (jsonify({"ok": True}) if ok
                    else jsonify({"ok": False, "error": "ptz_not_available"}), 200 if ok else 500)

    @app.route("/api/ptz/home", methods=["POST","OPTIONS"])
    @login_required
    def api_ptz_home():
        """
        Move PTZ to a known start pose.
        Body JSON (any subset):
          { "preset": "<id|name>" }  OR  { "pan": deg, "tilt": deg, "zoom": float }
          Optional: { "wait_settle": true, "settle_s": 0.8, "samples": 2, "sleep_s": 0.15 }
        Blocks manual control if auto-track is active (same as move_abs/rel).
        """
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        st = _read_ptz_status()
        if bool(st.get("enabled", False)):
            return jsonify({"ok": False, "error": "auto_track_active",
                            "message": "Auto-track is ON. Turn it OFF to home the PTZ manually."}), 409
        body = request.get_json(silent=True) or {}
        preset = (str(body.get("preset") or "").strip() or None)
        pan  = body.get("pan", None);   pan  = None if pan  is None else float(pan)
        tilt = body.get("tilt", None);  tilt = None if tilt is None else float(tilt)
        zoom = body.get("zoom", None);  zoom = None if zoom is None else float(zoom)
        wait_settle = bool(body.get("wait_settle", True))
        settle_s    = float(body.get("settle_s", 0.8))
        samples     = int(body.get("samples", 2))
        sleep_s     = float(body.get("sleep_s", 0.15))

        # Prefer the main worker (keeps behavior consistent with auto-track)
        try:
            from main import _ptz_q as _MAIN_PTZ_Q
            if preset:
                _MAIN_PTZ_Q.put_nowait({"type": "preset", "preset": preset, "ptz_cfg": _resolve_ptz_cfg_from_env()})
                # If caller also knows the nominal home pan/tilt, let UI sync the pose estimate
                if pan is not None and tilt is not None:
                    _MAIN_PTZ_Q.put_nowait({"type": "pose_estimate", "pan": pan, "tilt": tilt})
            else:
                _MAIN_PTZ_Q.put_nowait({"type": "abs", "pan": pan, "tilt": tilt, "zoom": zoom,
                                        "ptz_cfg": _resolve_ptz_cfg_from_env()})
                if pan is not None and tilt is not None:
                    _MAIN_PTZ_Q.put_nowait({"type": "pose_estimate", "pan": pan, "tilt": tilt})
            if wait_settle:
                _MAIN_PTZ_Q.put_nowait({"type": "settle", "timeout_s": settle_s,
                                        "samples": samples, "sleep_s": sleep_s})
            return jsonify({"ok": True})
        except Exception:
            # Fallback: direct control (best-effort), then bounded wait
            ok = False
            cfg = _resolve_ptz_cfg_from_env()
            if preset:
                ok = _ptz_direct_control({"type": "preset", "preset": preset, "ptz_cfg": cfg})
            else:
                ok = _ptz_direct_control({"type": "abs", "pan": pan, "tilt": tilt, "zoom": zoom, "ptz_cfg": cfg})
            if wait_settle and settle_s > 0:
                time.sleep(settle_s)
            return jsonify({"ok": bool(ok)}), (200 if ok else 500)

    @app.route("/api/ptz/pose", methods=["GET"])
    @login_required
    def api_ptz_pose():
        """
        Read current PTZ pose/speeds from the device (pan/tilt/zoom, pansp/tiltsp if exposed).
        """
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        if not _ptz_capable():
            return jsonify({"ok": False, "error": "PTZ not configured"}), 400
        cfg = _resolve_ptz_cfg_from_env()
        try:
            try:
                from ptz import PTZClient
            except Exception:
                from ptz_controller import PTZClient  # type: ignore
            client = PTZClient(
                cfg.get("host"),
                cfg.get("username") or cfg.get("user") or "",
                cfg.get("password") or cfg.get("pwd") or "",
                port=int(cfg.get("port") or 80),
                auth_mode=(cfg.get("auth_mode") or cfg.get("auth") or "auto"),
            )
            # Prefer parsed fields; include raw fallback for debugging
            pose = {}
            with contextlib.suppress(Exception):
                pose = client.status_parsed()
            if not pose:
                # raw text fallback
                txt = client.status()
                pose = {"raw": txt}
            return jsonify({"ok": True, "pose": pose})
        except Exception as e:
            return jsonify({"ok": False, "error": "unavailable", "details": str(e)}), 500

    @app.route("/api/ptz/home_config", methods=["GET","POST","OPTIONS"])
    @login_required
    def api_ptz_home_config():
        """
        GET  → return current ptz.home settings from config.json.
        POST → update ptz.home (pan_deg, tilt_deg, zoom, preset, on_enable, settle/wait_settle knobs).
        """
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        live = load_config() or {}
        ptz = live.get("ptz") or {}
        home = ptz.get("home") or {}

        if request.method == "GET":
            # Normalize legacy single fields into home{}
            resp = {
                "pan_deg":  float(home.get("pan_deg",  ptz.get("home_pan_deg",  0.0))),
                "tilt_deg": float(home.get("tilt_deg", ptz.get("home_tilt_deg", -5.0))),
                "zoom":     float(home.get("zoom",     ptz.get("home_zoom",     0.0))),
                "preset":   str(home.get("preset",     ptz.get("home_preset",   ""))),
                "on_enable": bool(home.get("on_enable", ptz.get("home_on_enable", True))),
                "settle_s":  float(home.get("settle_s", ptz.get("home_settle_s", 0.6))),
                "wait_settle": bool(home.get("wait_settle", ptz.get("home_wait_settle", True))),
                "settle_samples": int(home.get("settle_samples", ptz.get("settle_samples", 2))),
                "settle_sample_sleep_s": float(home.get("settle_sample_sleep_s", ptz.get("settle_sample_sleep_s", 0.15)))
            }
            return jsonify({"ok": True, "home": resp})

        # POST: update
        body = request.get_json(silent=True) or {}
        home_new = {
            "pan_deg":  float(body.get("pan_deg",  home.get("pan_deg",  0.0))),
            "tilt_deg": float(body.get("tilt_deg", home.get("tilt_deg", -5.0))),
            "zoom":     float(body.get("zoom",     home.get("zoom",     0.0))),
            "on_enable": bool(body.get("on_enable", home.get("on_enable", True))),
            "wait_settle": bool(body.get("wait_settle", home.get("wait_settle", True))),
            "settle_s": float(body.get("settle_s", home.get("settle_s", 0.6))),
            "settle_samples": int(body.get("settle_samples", home.get("settle_samples", 2))),
            "settle_sample_sleep_s": float(body.get("settle_sample_sleep_s", home.get("settle_sample_sleep_s", 0.15)))
        }
        # Optional preset (string or int → store as string)
        if body.get("preset") is not None:
            home_new["preset"] = str(body.get("preset")).strip()
        ptz["home"] = home_new
        live["ptz"] = ptz
        try:
            save_config(live)
            return jsonify({"ok": True, "home": home_new})
        except Exception as e:
            return jsonify({"ok": False, "error": "save_failed", "details": str(e)}), 500

    @app.route("/api/reload_config", methods=["POST"])
    @login_required
    def reload_config():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            # Write to a file flag or message queue
            with open("reload_flag.txt", "w") as f:
                f.write(str(time.time()))
            return jsonify({"status": "ok", "message": "Config reload requested."})
        except Exception as e:
            logger.error(f"[RELOAD API ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500

    @app.route("/manual_snapshot", methods=["POST"])
    @login_required
    def manual_snapshot():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403

        try:
            now = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            config = load_config()
            selected = config.get("selected_camera", 0)
            cam = config.get("cameras", [{}])[selected] if isinstance(config.get("cameras"), list) else {}
            # Best-effort radar context; interface may not expose get_latest_frame()
            objects = []
            try:
                if hasattr(radar, "get_targets"):
                    objects = radar.get_targets() or []
                elif hasattr(radar, "get_latest_frame"):
                    rf = radar.get_latest_frame() or {}
                    objects = rf.get("objects", []) or []
            except Exception as _e:
                logger.debug(f"[CALIB] radar fallback non-fatal: {_e}")

            snapshot_url = cam.get("snapshot_url", cam.get("url"))
            username = cam.get("username")
            password = cam.get("password")
            auth = HTTPDigestAuth(username, password) if username and password else None
            response = requests.get(snapshot_url, auth=auth, timeout=5)

            if response.status_code != 200 or not response.content.startswith(b'\xff\xd8'):
                return jsonify({"error": "Snapshot capture failed"}), 500

            snapshot_path = os.path.join(SNAPSHOT_FOLDER, f"manual_{timestamp}.jpg")
            with open(snapshot_path, "wb") as f:
                f.write(response.content)

            label = f"MANUAL | {datetime.now().strftime('%H:%M:%S')}"
            conf_thresh = config.get("annotation_conf_threshold", 0.5)
            annotated_path, visual_distance, corrected_distance, bbox, yolo_label = annotate_speeding_object(
                image_path=snapshot_path,
                radar_distance=0.0,  # legacy param, ignored internally
                label=label,
                min_confidence=conf_thresh
            )
            # Optional ANPR on manual snapshot (if a vehicle)
            plate_text = None
            plate_conf = 0.0
            plate_bbox = None
            plate_crop_path = None
            try:
                type_hint = (yolo_label or "").upper()
                if bbox and any(k in type_hint for k in ("CAR","BUS","TRUCK","BIKE","BICYCLE","VEHICLE")):
                    # Use the original frame to preserve plate area; keep bbox as ROI hint.
                    anpr = run_anpr(snapshot_path, roi=bbox, save_dir=SNAPSHOT_FOLDER) or {}
                    plate_text = anpr.get("plate_text")
                    plate_conf = float(anpr.get("plate_conf", 0.0))
                    plate_bbox = anpr.get("plate_bbox")
                    plate_crop_path = anpr.get("crop_path")
            except Exception as _e:
                logger.debug(f"[ANPR manual] {__name__}: {_e}")

            if not annotated_path:
                return jsonify({"error": "Annotation failed"}), 500

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                # Pick most relevant object if available; otherwise synthesize a minimal record
                radar_obj = (objects[0] if objects else {})
                if yolo_label:
                    radar_obj["type"] = yolo_label

                cursor.execute("""
                    INSERT INTO radar_data (
                        timestamp, datetime, sensor, object_id, type, confidence, speed_kmh,
                        velocity, distance, direction, signal_level, doppler_frequency, snapshot_path,
                        x, y, z, range, azimuth, elevation, motion_state, snapshot_status,
                        velx, vely, velz, snr, noise,
                        reviewed, flagged, range_profile, noise_profile,
                        accx, accy, accz, snapshot_type,
                        plate_text, plate_conf, plate_bbox, plate_crop_path
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s
                            %s, %s, %s, %s)
                """, (
                    now,
                    datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                    "Manual",
                    f"manual_{uuid.uuid4().hex[:6]}",
                    radar_obj.get("type", "UNKNOWN"),
                    radar_obj.get("confidence", 0.0),
                    radar_obj.get("speed_kmh", 0.0),
                    radar_obj.get("velocity", 0.0),
                    radar_obj.get("distance", 0.0),
                    radar_obj.get("direction", "manual"),
                    radar_obj.get("signal_level", 0.0),
                    radar_obj.get("doppler_frequency", 0.0),
                    annotated_path,
                    radar_obj.get("x", 0.0),
                    radar_obj.get("y", 0.0),
                    radar_obj.get("z", 0.0),
                    radar_obj.get("range", 0.0),
                    radar_obj.get("azimuth", 0.0),
                    radar_obj.get("elevation", 0.0),
                    radar_obj.get("motion_state", "STATIC"),
                    "valid",
                    radar_obj.get("velx", 0.0),
                    radar_obj.get("vely", 0.0),
                    radar_obj.get("velz", 0.0),
                    radar_obj.get("snr", 0.0),
                    radar_obj.get("noise", 0.0),
                    0, 0,
                    radar_obj.get("range_profile", []),
                    radar_obj.get("noise_profile", []),
                    radar_obj.get("accx", 0.0),
                    radar_obj.get("accy", 0.0),
                    radar_obj.get("accz", 0.0),
                    "manual",
                    plate_text,
                    plate_conf,
                    Json(plate_bbox if plate_bbox else []),
                    plate_crop_path
                ))
                conn.commit()

            return jsonify({"status": "ok", "message": "Snapshot captured successfully."})

        except Exception as e:
            logger.error(f"[MANUAL SNAPSHOT ERROR] {e}")
            return jsonify({"error": "Internal error"}), 500

    @app.route("/logs")
    @login_required
    def view_logs():
        """
        Render logs page with a selectable source; default to 'radar'.
        """
        try:
            # choose default or query param
            selected = (request.args.get("name") or "radar").strip()
            if selected not in LOG_SOURCES:
                selected = "radar"
            path, _label = LOG_SOURCES[selected]
            logs = []
            if os.path.isfile(path):
                logs = _tail_file(path, 1000)  # first render shows last 1000
            else:
                logs = [f"[INFO] Log file not found: {path}"]

            # build sources for the UI selector
            sources = [{"name": k, "label": v[1]} for k, v in LOG_SOURCES.items()]
            sources.sort(key=lambda x: x["label"].lower())

            # flag for CSV (front-end may style differently if desired)
            is_csv = path.endswith(".csv")
            current_label = _label
            return render_template("logs.html",
                                   logs=logs,
                                   sources=sources,
                                   selected_name=selected,
                                   current_label=current_label,
                                   is_csv=is_csv)
        except Exception as e:
            import traceback
            err_msg = f"Exception in /logs: {e}\n{traceback.format_exc()}"
            return f"<pre style='color:red;'>{err_msg}</pre>", 500

    @app.route("/api/logs")
    @login_required
    def api_logs():
        """
        Paginated log reader.
        GET params:
          - name: one of LOG_SOURCES keys
          - offset, limit: pagination (lines)
        """
        try:
            name = (request.args.get("name") or "radar").strip()
            if name not in LOG_SOURCES:
                return jsonify({"error": "unknown log name"}), 400
            path, _ = LOG_SOURCES[name]
            offset = max(0, int(request.args.get("offset", 0)))
            limit  = max(1, min(2000, int(request.args.get("limit", 100))))
            max_lines = offset + limit

            if not os.path.exists(path):
                return jsonify({"logs": [], "has_more": False})

            # read tail of file up to max_lines; CSV is treated as text lines
            all_lines = _tail_file(path, max_lines)
            paginated = all_lines[-limit:]

            return jsonify({
               "logs": [line.rstrip("\n") for line in paginated],
                "has_more": len(all_lines) >= max_lines,
                "is_csv": path.endswith(".csv"),
            })
        except Exception as e:
            logger.exception(f"[LOGS API ERROR] {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/logs/download")
    @login_required
    def download_log():
        """
        Download the currently selected log (full file).
        """
        try:
            name = (request.args.get("name") or "radar").strip()
            if name not in LOG_SOURCES:
                return "unknown log name", 400
            path, label = LOG_SOURCES[name]
            if not os.path.isfile(path):
                return "file not found", 404
            as_name = os.path.basename(path)
            return send_file(path, as_attachment=True, download_name=as_name)
        except Exception as e:
            return f"download error: {e}", 500

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"].strip()
            password = request.form["password"]
            user = get_user_by_username(username)
            if user and check_password_hash(user.password_hash, password):
                remember_me = bool(request.form.get("remember"))
                login_user(user, remember=remember_me)
                try:
                    update_user_activity(user.id)
                except Exception:
                    pass
                nxt = request.form.get("next") or request.args.get("next")
                return redirect(nxt or url_for("index"))
            return render_template("login.html", error="Invalid credentials", prefill_username=username)
        return render_template("login.html")


    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out successfully", "success")
        return redirect(url_for("login"))

    @app.route("/cameras", methods=["GET"])
    @login_required
    def cameras_grid():
        # Pull only active cameras for the grid
        cams = _active_cameras_from_db()
        return render_template("cameras.html", cams=cams)

    @app.route("/")
    @login_required
    def index():
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # Get recent detections 
                cursor.execute("""
                    SELECT measured_at, type, speed_kmh, distance, direction, motion_state,
                           snapshot_type, snapshot_path, object_id, confidence, azimuth, elevation,
                           plate_text, plate_conf, plate_crop_path
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
                    ORDER BY measured_at DESC
                    LIMIT 10
                """)
                rows = cursor.fetchall()

                # Get summary statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN LOWER(COALESCE(type, '')) LIKE '%human%' OR LOWER(COALESCE(type, '')) LIKE '%person%' THEN 1 ELSE 0 END) as humans,
                        SUM(CASE WHEN LOWER(COALESCE(type, '')) LIKE '%vehicle%' OR LOWER(COALESCE(type, '')) LIKE '%car%' THEN 1 ELSE 0 END) as vehicles,
                        AVG(CASE WHEN speed_kmh IS NOT NULL AND speed_kmh >= 0 THEN speed_kmh END) as avg_speed,
                        MAX(measured_at) as last_detection
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
                """)
                stats = cursor.fetchone()

                total, humans, vehicles, avg_speed, last_detection = (
                    stats['total'] if stats else 0,
                    stats['humans'] if stats else 0,
                    stats['vehicles'] if stats else 0,
                    stats['avg_speed'] if stats and stats['avg_speed'] is not None else 0,
                    stats['last_detection'] if stats else None
                )

                # Build snapshot card data (normalized for legacy + bundle)
                snapshots = []
                for r in rows:
                    # Always make sure /snapshots/<filename> is resolvable,
                    # even when DB path points inside a bundle/image.jpg
                    fname = _ensure_public_snapshot_filename(r.get("snapshot_path") or "")
                    if fname.lower().endswith("_plate.jpg"):
                        continue

                    speed        = float(r.get("speed_kmh") or 0)
                    dist_m       = float(r.get("distance") or 0)
                    conf         = float(r.get("confidence") or 0)
                    clip_path    = r.get("clip_path")
                    clip_status  = (r.get("clip_status") or "").lower() if r.get("clip_status") else None
                    clip_ready   = bool(clip_path and clip_status == "ready" and os.path.exists(clip_path))
                    clip_file    = os.path.basename(clip_path) if clip_path else None
                    # Provide a clip URL for the card even if DB still says pending
                    clip_url = None
                    if clip_ready and clip_file:
                        clip_url = url_for("serve_clip", filename=clip_file)
                    else:
                        # Fallback to bundle/clip.mp4 if it exists
                        bundle_dir = _guess_bundle_from_snapshot_path(r.get("snapshot_path") or "")
                        if bundle_dir:
                            bundle_clip = os.path.join(bundle_dir, "clip.mp4")
                            if os.path.isfile(bundle_clip):
                                clip_ready  = True
                                clip_status = "ready"      # presentational override for the card
                                clip_file   = "clip.mp4"
                                clip_url    = url_for("bundle_asset", snapshot=fname, name="clip.mp4")

                    # Bundle context (if present)
                    bundle_dir   = _guess_bundle_from_snapshot_path(r.get("snapshot_path") or "")
                    bundle_media = []
                    main_thumb   = None
                    if bundle_dir and os.path.isdir(bundle_dir):
                        man = _bundle_manifest(bundle_dir)
                        main_thumb = man.get("image") or None
                        # collect bundle images
                        for nm in (man.get("images") or []):
                            bundle_media.append({
                                "name": nm, "type": "image",
                                "url": url_for("bundle_asset", snapshot=fname, name=nm)
                            })
                        # collect bundle videos
                        for nm in (man.get("videos") or []):
                            bundle_media.append({
                                "name": nm, "type": "video",
                                "url": url_for("bundle_asset", snapshot=fname, name=nm)
                            })

                    # Legacy plate crop normalize to basename under /snapshots
                    plate_crop_fname = None
                    _plate = r.get("plate_crop_path") or None
                    if _plate:
                        try:
                            base = os.path.basename(str(_plate))
                            if os.path.isfile(os.path.join(SNAPSHOT_FOLDER, base)):
                                plate_crop_fname = base
                                # Put plate first in the media strip
                                bundle_media.insert(0, {
                                    "name": base, "type": "image",
                                    "url": url_for("serve_snapshot", filename=base)
                                })
                        except Exception:
                            plate_crop_fname = None

                    # Fallbacks for main thumbnail
                    if not main_thumb:
                        # Prefer the normalized /snapshots/<fname> (works for legacy)
                        main_thumb = fname

                    # Build a guaranteed-good URL for the main thumbnail
                    try:
                        if bundle_dir and os.path.isdir(bundle_dir):
                            man = man if 'man' in locals() else _bundle_manifest(bundle_dir)
                            if main_thumb == (man.get("image") or main_thumb):
                                main_thumb_url = url_for("bundle_asset", snapshot=fname, name=main_thumb)
                            else:
                                main_thumb_url = url_for("serve_snapshot", filename=main_thumb)
                        else:
                            main_thumb_url = url_for("serve_snapshot", filename=main_thumb)
                    except Exception:
                        main_thumb_url = url_for("serve_snapshot", filename=main_thumb)

                    snapshots.append({
                        "filename": fname or "no_image.jpg",
                        "datetime": to_ist(r.get("measured_at") or r.get("datetime")),
                        "type": (r.get("type") or "UNKNOWN"),
                        "speed": round(speed, 2),
                        "distance": round(dist_m, 2),
                        "direction": (r.get("direction") or "N/A"),
                        "azimuth": round(float(r["azimuth"]) if r.get("azimuth") is not None else 0, 2),
                        "elevation": round(float(r["elevation"]) if r.get("elevation") is not None else 0, 2),
                        "motion_state": (r.get("motion_state") or "N/A"),
                        "object_id": (r.get("object_id") or "N/A"),
                        "confidence": round(conf, 2),
                        "snapshot_type": (r.get("snapshot_type") or "auto"),
                        "snapshot_status": (r.get("snapshot_status") or "").lower(),
                        "reviewed": r.get("reviewed") or 0,
                        "flagged": r.get("flagged") or 0,
                        "plate": (r.get("plate_text") or None),
                        "plate_conf": (float(r["plate_conf"]) if r.get("plate_conf") is not None else None),
                        "plate_image": plate_crop_fname,
                        "plate_crop": plate_crop_fname,
                        "clip_status": clip_status,
                        "clip_ready": clip_ready,
                        "clip_url": (url_for("serve_clip", filename=clip_file) if clip_ready and clip_file else None),
                        "clip_gdrive_link": r.get("clip_gdrive_link"),
                        "bundle_media": bundle_media,
                        "main_thumb": main_thumb,               # keep basename for back-compat
                        "main_thumb_url": main_thumb_url        
                    })

            # Load logs
            log_path = os.path.join("system-logs", "radar.log")
            logs = []
            if os.path.isfile(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    logs = f.readlines()[-15:]
            else:
                logs = ["[INFO] Log file not found: system-logs/radar.log"]
            
            last_detection_str = to_ist(last_detection) if last_detection else "N/A"

            # Pi health
            pi = _fetch_pi_health()
            temp_c = pi.get("temperature")
            cpu_load_val = pi.get("cpu_load") if isinstance(pi.get("cpu_load"), (int, float)) else 0.0

            # Build summary
            summary = {
                "total": total or 0,
                "humans": humans or 0,
                "vehicles": vehicles or 0,
                "average_speed": round(avg_speed, 2) if avg_speed else 0,
                "last_detection": last_detection_str,
                "logs": logs,
                "pi_temperature": round(temp_c, 1) if isinstance(temp_c, (int, float)) else None,
                "cpu_load": round(float(cpu_load_val or 0.0), 1),
                "radar_connected": bool(pi.get("radar_connected")) if pi.get("ok") else None
            }

            return render_template("index.html", snapshots=snapshots, summary=summary, config=load_config())

        except Exception as e:
            logger.error(f"[INDEX ROUTE ERROR] {e}")
            flash("Error loading dashboard data", "error")
            summary = {
                "total": 0,
                "humans": 0,
                "vehicles": 0,
                "average_speed": 0,
                "last_detection": "N/A",
                "logs": [],
                "pi_temperature": "N/A",
                "cpu_load": 0.0
            }
            return render_template("index.html", snapshots=[], summary=summary, config=load_config())
        
    @app.route("/api/charts")
    def api_charts():
        try:
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                try:
                    days = int(request.args.get("days", 30))
                except ValueError:
                    days = 30

                if days <= 0:
                    date_filter = "DATE((measured_at AT TIME ZONE 'Asia/Kolkata')) = DATE(NOW() AT TIME ZONE 'Asia/Kolkata')"
                    params = ()
                else:
                    date_filter = "measured_at >= (NOW() - make_interval(days => %s))"
                    params = (days,)
                hour_expr = "TO_CHAR((measured_at AT TIME ZONE 'Asia/Kolkata'),'HH24')"

                # Speed histogram
                cur.execute(f"""
                    SELECT speed_kmh
                    FROM radar_data
                    WHERE speed_kmh IS NOT NULL AND speed_kmh BETWEEN 0 AND 200
                        AND snapshot_path IS NOT NULL AND snapshot_path <> ''
                    AND {date_filter}
                """, params)
                speeds = [float(r["speed_kmh"]) for r in cur.fetchall()]
                bins   = list(range(0, 101, 10))
                labels = [f"{i}-{i+9}" for i in bins[:-1]] + ["100+"]
                counts = [0]*len(labels)
                for s in speeds:
                    idx = int(s//10) if s < 100 else -1
                    counts[idx] += 1

                # Direction breakdown (normalize)
                cur.execute(f"""
                    SELECT direction
                    FROM radar_data
                    WHERE direction IS NOT NULL AND TRIM(direction) <> ''
                        AND snapshot_path IS NOT NULL AND snapshot_path <> ''
                    AND {date_filter}
                """, params)
                raw = [str(r["direction"]).strip().lower() for r in cur.fetchall()]
                def norm(d):
                    if d in ("towards","approaching"): return "Approaching"
                    if d in ("away","departing"):       return "Departing"
                    if d in ("static","stationary"):    return "Stationary"
                    if d == "left":                     return "Left"
                    if d == "right":                    return "Right"
                    return "Unknown"
                dir_labels = ["Approaching","Departing","Stationary","Left","Right","Unknown"]
                dir_counts = [0]*len(dir_labels)
                for d in raw:
                    dir_counts[dir_labels.index(norm(d))] += 1

                # Violations per hour (always produce 24 buckets)
                cur.execute(f"""
                    SELECT {hour_expr} AS hour, COUNT(*) AS count
                    FROM radar_data
                    WHERE COALESCE(speed_kmh,0) > 0
                    AND {date_filter}
                GROUP BY hour
                ORDER BY hour
                """, params)
                rows = cur.fetchall()
                hour_map   = {int(r['hour']): r['count'] for r in rows}
                hour_labels = [f"{h:02d}:00" for h in range(24)]
                hour_data   = [hour_map.get(h, 0) for h in range(24)]

                return jsonify({
                    "speed_histogram":     {"labels": labels, "data": counts},
                    "direction_breakdown": {"labels": dir_labels, "data": dir_counts},
                    "violations_per_hour": {"labels": hour_labels, "data": hour_data},
                })
        except psycopg2.Error:
            logger.exception("[API CHARTS] PostgreSQL error")
            return jsonify({"error": "Database error"}), 500
        except Exception:
            logger.exception("[API CHARTS] Unhandled error")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/gallery")
    @login_required
    def gallery():
        # Query params
        obj_type        = (request.args.get("type") or "").strip().upper()
        min_speed       = float(request.args.get("min_speed") or 0)
        max_speed       = float(request.args.get("max_speed") or 999)
        direction       = (request.args.get("direction") or "").strip().lower()
        motion_state    = (request.args.get("motion_state") or "").strip().lower()
        object_id       = (request.args.get("object_id") or "").strip()
        start_date      = (request.args.get("start_date") or "").strip()
        end_date        = (request.args.get("end_date") or "").strip()
        min_confidence  = float(request.args.get("min_confidence") or 0)
        max_confidence  = float(request.args.get("max_confidence") or 1)
        snapshot_type   = (request.args.get("snapshot_type") or "").strip().lower()
        reviewed_only   = request.args.get("reviewed_only") == "1"
        flagged_only    = request.args.get("flagged_only") == "1"
        unannotated_only= request.args.get("unannotated_only") == "1"
        download_zip    = request.args.get("download") == "1"
        selected_raw    = (request.args.get("selected") or "").strip()

        page  = max(int(request.args.get("page", 1)), 1)
        limit = min(max(int(request.args.get("limit", 100)), 1), 1000)
        offset = (page - 1) * limit

        try:
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                where = ["snapshot_path IS NOT NULL", "snapshot_path <> ''"]
                params = []

                # Selected basenames (comma-separated)
                if selected_raw:
                    selected_list = [os.path.basename(s) for s in selected_raw.split(",") if s.strip()]
                    if selected_list:
                        ph = ", ".join(["%s"] * len(selected_list))
                        where.append(f"regexp_replace(snapshot_path, '^.*[\\\\/]', '') IN ({ph})")
                        params += selected_list

                if min_speed > 0 or max_speed < 999:
                    where.append("speed_kmh BETWEEN %s AND %s")
                    params += [min_speed, max_speed]
                if obj_type:
                    where.append("UPPER(COALESCE(type,'')) LIKE %s")
                    params.append(f"%{obj_type}%")
                if direction:
                    where.append("LOWER(COALESCE(direction,'')) LIKE %s")
                    params.append(f"%{direction}%")
                if motion_state:
                    where.append("LOWER(COALESCE(motion_state,'')) LIKE %s")
                    params.append(f"%{motion_state}%")
                if object_id:
                    where.append("COALESCE(object_id,'') LIKE %s")
                    params.append(f"%{object_id}%")
                if start_date:
                    where.append("DATE(measured_at) >= %s")
                    params.append(start_date)
                if end_date:
                    where.append("DATE(measured_at) <= %s")
                    params.append(end_date)
                if min_confidence > 0 or max_confidence < 1:
                    where.append("confidence BETWEEN %s AND %s")
                    params += [min_confidence, max_confidence]
                if snapshot_type in ("manual", "auto"):
                    where.append("snapshot_type = %s")
                    params.append(snapshot_type)
                if reviewed_only:
                    where.append("reviewed = 1")
                if flagged_only:
                    where.append("flagged = 1")
                if unannotated_only:
                    where.append("reviewed = 0 AND flagged = 0")

                where_sql = " AND ".join(where)

                # Count total for pagination
                cur.execute(f"SELECT COUNT(*) AS c FROM radar_data WHERE {where_sql}", params)
                total_items = int(cur.fetchone()["c"])
                total_pages = max((total_items + limit - 1) // limit, 1)

                # Fetch page (stay inside the same DB context)
                cur.execute(f"""
                SELECT
                    measured_at, datetime, type, speed_kmh, distance, direction,
                    snapshot_path, object_id, confidence, reviewed, flagged,
                    motion_state, snapshot_type, snapshot_status, azimuth, elevation,
                    clip_path, clip_status, clip_duration_s, clip_fps, clip_size_bytes, clip_gdrive_link,
                    plate_text, plate_conf, plate_crop_path
                    FROM radar_data
                    WHERE {where_sql}
                    ORDER BY measured_at DESC
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])
                rows = cur.fetchall()

            # Prepare card data
            snapshots = []
            for r in rows:
                # Normalize filename so /snapshots/<name> always exists (handles bundle/image.jpg)
                fname   = _ensure_public_snapshot_filename(r.get("snapshot_path") or "")
                if fname.lower().endswith("_plate.jpg"):
                    continue
                speed   = float(r.get("speed_kmh") or 0)
                dist_m  = float(r["distance"] or 0)
                conf    = float(r.get("confidence") or 0)
                clip_path   = r.get("clip_path")
                clip_status = (r.get("clip_status") or "").lower() if r.get("clip_status") else None
                clip_ready  = bool(clip_path and clip_status == "ready" and os.path.exists(clip_path))
                clip_file   = os.path.basename(clip_path) if clip_path else None

                bundle_dir      = _guess_bundle_from_snapshot_path(r.get("snapshot_path") or "")
                bundle_open_url = url_for("bundle_open", snapshot=fname) if bundle_dir else None
                bundle_zip_url  = url_for("bundle_zip",  snapshot=fname) if bundle_dir else None
                # Fallback: if DB says pending but bundle has clip.mp4, use it for playback
                clip_url = None
                clip_filename = clip_file
                if clip_ready:
                    clip_url = url_for("serve_clip", filename=clip_file) if clip_file else None
                elif bundle_dir:
                    bundle_clip = os.path.join(bundle_dir, "clip.mp4")
                    if os.path.isfile(bundle_clip):
                        clip_ready = True
                        clip_filename = "clip.mp4"
                        clip_url = url_for("bundle_asset", snapshot=fname, name="clip.mp4")

                # --- normalize plate crop to a plain filename under SNAPSHOT_FOLDER
                plate_crop_fname = None
                _plate_crop_path = r.get("plate_crop_path") or None
                if _plate_crop_path:
                    try:
                        base = os.path.basename(str(_plate_crop_path))
                        if os.path.isfile(os.path.join(SNAPSHOT_FOLDER, base)):
                            plate_crop_fname = base
                    except Exception:
                        plate_crop_fname = None  # be safe if anything odd happens

                snapshots.append({
                    "filename": fname or "no_image.jpg",
                    "datetime": to_ist(r.get("measured_at") or r.get("datetime")),
                    "type": (r["type"] or "UNKNOWN"),
                    "speed": round(speed, 2),
                    "distance": round(dist_m, 2),
                    "direction": (r["direction"] or "N/A"),
                    "object_id": (r["object_id"] or "N/A"),
                    "confidence": round(conf, 2),
                    "reviewed": r["reviewed"] or 0,
                    "flagged": r["flagged"] or 0,
                    "motion_state": (r["motion_state"] or "N/A"),
                    "snapshot_type": (r["snapshot_type"] or "auto"),
                    "snapshot_status": (r["snapshot_status"] or "valid"),
                    "azimuth": 0.0 if r["azimuth"] is None else float(r["azimuth"]),
                    "elevation": 0.0 if r["elevation"] is None else float(r["elevation"]),
                    "path": r["snapshot_path"] if r["snapshot_path"] and os.path.exists(r["snapshot_path"]) else None,
                    "label": f"{(r['type'] or 'UNKNOWN')} | {speed:.2f} km/h | {dist_m:.2f} m | {(r['direction'] or 'N/A')}",
                    "clip_status": clip_status,
                    "clip_ready": bool(clip_ready),
                    "clip_filename": clip_filename,
                    "clip_url": clip_url,
                    "clip_gdrive_link": (r.get("clip_gdrive_link") or ""),
                    "bundle_open_url": bundle_open_url,
                    "bundle_zip_url": bundle_zip_url,
                    "plate": (r.get("plate_text") or None),
                    "plate_conf": (float(r.get("plate_conf")) if r.get("plate_conf") is not None else None),
                    "plate_crop": plate_crop_fname,
                })

            # ZIP for current filter result
            if download_zip and snapshots:
                buf = BytesIO()
                with zipfile.ZipFile(buf, "w") as zipf:
                    for s in snapshots:
                        if s["path"]:
                            try:
                                zipf.write(s["path"], arcname=s["filename"])
                            except Exception as e:
                                logger.warning(f"[ZIP] Skip {s['filename']}: {e}")
                        # include clip if present/ready
                        try:
                            if s.get("clip_ready") and s.get("clip_filename"):
                                clip_fp = os.path.join(CLIPS_FOLDER, s["clip_filename"])
                                if os.path.exists(clip_fp):
                                    zipf.write(clip_fp, arcname=s["clip_filename"])
                        except Exception as e:
                            logger.warning(f"[ZIP] Skip clip for {s.get('filename')}: {e}")
                buf.seek(0)
                return send_file(buf, mimetype="application/zip", as_attachment=True,
                                download_name="filtered_snapshots.zip")

            return render_template("gallery.html",
                                snapshots=snapshots,
                                current_page=page,
                                total_pages=total_pages)

        except Exception as e:
            logger.exception("[GALLERY] error")
            flash("Error loading gallery data", "error")
            return render_template("gallery.html", snapshots=[], current_page=1, total_pages=1)

    @app.route("/mark_snapshot", methods=["POST"])
    @login_required
    def mark_snapshot():
        try:
            data = request.get_json()
            snapshot = data.get("snapshot")
            action = data.get("action") 

            if not snapshot or action not in ("reviewed", "flagged"):
                return jsonify({"error": "Invalid input"}), 400

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute(f"SELECT {action} FROM radar_data WHERE snapshot_path LIKE %s", (f"%{snapshot}",))
                current = cursor.fetchone()
                new_value = 0 if current and current[action] == 1 else 1

                cursor.execute(f"""
                    UPDATE radar_data SET {action} = %s 
                    WHERE snapshot_path LIKE %s
                """, (new_value, f"%{snapshot}",))
                conn.commit()

            return jsonify({"status": "updated", "new_value": new_value})
        
        except Exception as e:
            logger.error(f"Error marking snapshot: {e}")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route("/delete_snapshot/<path:filename>", methods=["DELETE"])
    @login_required
    def delete_snapshot(filename):
        if not is_admin():
            return jsonify({"success": False, "error": "Unauthorized"}), 403
        try:
            # Resolve current DB row (latest) to get absolute file paths first
            snap_fp = None
            with get_db_connection() as conn:
                row = _row_by_snapshot_basename(conn, filename)
                if row:
                    snap_fp = (row.get("snapshot_path") or "").strip()
                    # If there is a clip linked, remove the files and clear DB clip_* fields
                    if row.get("clip_path"):
                        _delete_clip_files(row)
                        _update_clip_fields_by_basename(
                            conn, filename,
                            clip_status="deleted",
                            clip_path=None,
                            clip_duration_s=None,
                            clip_fps=None,
                            clip_size_bytes=None,
                            clip_sha256=None,
                            clip_gdrive_link=None
                        )
                # Delete DB rows by basename
                cur = conn.cursor()
                cur.execute("""
                    DELETE FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
                      AND regexp_replace(snapshot_path, '^.*[\\\/]', '') = %s
                """, (filename,))
                deleted = cur.rowcount
                conn.commit()
            # Delete the image file from disk (DB might have stored absolute path)
            if not snap_fp:
                # fall back to conventional path if DB lookup failed
                snap_fp = os.path.join(SNAPSHOT_FOLDER, filename)
            _safe_unlink(snap_fp)
            return jsonify({"success": True, "deleted": int(deleted)})
        except Exception as e:
            logger.error(f"[DELETE SNAPSHOT ERROR] {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/gallery/delete", methods=["POST"])
    @login_required
    def api_gallery_delete():
        """
        Delete detections by scope:
        - scope='selected' : only listed basenames (intersect with current filters)
        - scope='filtered' : everything matching current filters
        - scope='all'      : ALL detections (ignores filters)
        Body: { scope, filters: {...}, selected: "a.jpg,b.jpg" }
        """
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            body = request.get_json(silent=True) or {}
            scope = (body.get("scope") or "filtered").strip().lower()
            filters = (body.get("filters") or {}) if scope != "all" else {}
            selected_raw = (body.get("selected") or "").strip()

            basename_sql = "regexp_replace(snapshot_path, '^.*[\\\\/]', '')"
            where = ["snapshot_path IS NOT NULL", "snapshot_path <> ''", f"{basename_sql} NOT ILIKE %s"]
            params = ['%\\_plate.jpg']  # exclude plate crops everywhere

            # Selected subset
            if scope == "selected" and selected_raw:
                sels = [os.path.basename(s) for s in selected_raw.split(",") if s.strip()]
                if not sels:
                    return jsonify({"error": "No selected items"}), 400
                placeholders = ", ".join(["%s"] * len(sels))
                where.append(f"{basename_sql} IN ({placeholders})")
                params.extend(sels)

            # Apply same filtering semantics as /gallery
            if filters:
                t = (filters.get("type") or "").strip().upper()
                if t:
                    where.append("UPPER(COALESCE(type,'')) LIKE %s"); params.append(f"%{t}%")
                if filters.get("min_speed") or filters.get("max_speed"):
                    min_s = float(filters.get("min_speed") or 0); max_s = float(filters.get("max_speed") or 999)
                    where.append("speed_kmh BETWEEN %s AND %s"); params += [min_s, max_s]
                d = (filters.get("direction") or "").strip().lower()
                if d:
                    where.append("LOWER(COALESCE(direction,'')) LIKE %s"); params.append(f"%{d}%")
                ms = (filters.get("motion_state") or "").strip().lower()
                if ms:
                    where.append("LOWER(COALESCE(motion_state,'')) LIKE %s"); params.append(f"%{ms}%")
                oid = (filters.get("object_id") or "").strip()
                if oid:
                    where.append("COALESCE(object_id,'') LIKE %s"); params.append(f"%{oid}%")
                sd = (filters.get("start_date") or "").strip()
                if sd:
                    where.append("DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) >= %s"); params.append(sd)
                ed = (filters.get("end_date") or "").strip()
                if ed:
                    where.append("DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) <= %s"); params.append(ed)
                if filters.get("min_confidence") or filters.get("max_confidence"):
                    min_c = float(filters.get("min_confidence") or 0); max_c = float(filters.get("max_confidence") or 1)
                    where.append("confidence BETWEEN %s AND %s"); params += [min_c, max_c]
                st = (filters.get("snapshot_type") or "").strip().lower()
                if st in ("manual", "auto"):
                    where.append("LOWER(COALESCE(snapshot_type,'')) = %s"); params.append(st)
                if str(filters.get("reviewed_only") or "") == "1":
                    where.append("reviewed = 1")
                if str(filters.get("flagged_only") or "") == "1":
                    where.append("flagged = 1")
                if str(filters.get("unannotated_only") or "") == "1":
                    where.append("reviewed = 0 AND flagged = 0")

            where_sql = " AND ".join(where) if scope != "all" else "TRUE"

            # Collect rows to delete files, then delete DB rows
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute(f"SELECT snapshot_path, plate_crop_path, clip_path FROM radar_data WHERE {where_sql}", params if scope != "all" else [])
                rows = cur.fetchall() or []
                _delete_files_for_rows(rows)
                cur.execute(f"DELETE FROM radar_data WHERE {where_sql}", params if scope != "all" else [])
                deleted = cur.rowcount
                conn.commit()

            return jsonify({"ok": True, "deleted": int(deleted), "scope": scope})
        except Exception as e:
            logger.exception(f"[API GALLERY DELETE] {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/clip_delete", methods=["POST"])
    @login_required
    def api_clip_delete():
        """
        Delete only the video clip (and its .json) linked to a snapshot.
        Body: { "snapshot": "<basename.jpg>" }
        Also clears clip_* columns in the matching row.
        """
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            data = request.get_json(silent=True) or {}
            snap = (data.get("snapshot") or "").strip()
            if not snap:
                return jsonify({"error": "snapshot required"}), 400
            with get_db_connection() as conn:
                row = _row_by_snapshot_basename(conn, snap)
                if not row:
                    return jsonify({"error": "not found"}), 404
                _delete_clip_files(row)
                _update_clip_fields_by_basename(
                    conn, snap,
                    clip_status="deleted",
                    clip_path=None,
                    clip_duration_s=None,
                    clip_fps=None,
                    clip_size_bytes=None,
                    clip_sha256=None,
                    clip_gdrive_link=None
                )
            return jsonify({"status": "ok"})
        except Exception as e:
            logger.exception("[CLIP DELETE] error")
            return jsonify({"error": "Internal server error"}), 500

    # ─────────────────────────────────────────────────────────────────────────────
    # Camera test tool (probes ALL active cameras)
    # ─────────────────────────────────────────────────────────────────────────────
    @app.route("/cam_test", methods=["GET"])
    @login_required
    def cam_test():
        cams = _active_cameras_from_db()
        results = []
        for cam in cams:
            ok = _probe_camera(cam, timeout_s=float(config.get("camera_timeout", 5.0)))
            results.append({
                "id": cam.get("id"),
                "name": cam.get("name"),
                "ok": bool(ok),
                "stream_type": cam.get("stream_type"),
                "url": cam.get("url") or cam.get("snapshot_url")
            })
        return jsonify({"count": len(results), "results": results})

    # ─────────────────────────────────────────────────────────────────────────────
    # Camera frame & MJPEG proxy (works for RTSP/MJPEG/snapshot)
    # ─────────────────────────────────────────────────────────────────────────────
    def _mjpeg_generator(cam: dict, interval_s: float = 0.4):
        boundary = b"--frame\r\n"
        while True:
            try:
                jpg = _grab_one_jpeg(cam, timeout_s=6.0)
                if jpg:
                    yield boundary
                    yield b"Content-Type: image/jpeg\r\n"
                    yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                    yield jpg
                    yield b"\r\n"
                time.sleep(interval_s)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.6)

    @app.route("/cam_mjpeg/<int:cam_id>")
    @login_required
    def cam_mjpeg(cam_id: int):
        cams, _ = load_cameras_from_db()
        cam = next((c for c in cams if int(c.get("id", -1)) == int(cam_id)), None)
        if not cam or not cam.get("enabled", True):
            abort(404)
        resp = Response(_mjpeg_generator(cam), mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/cam_frame/<int:cam_id>")
    @login_required
    def cam_frame(cam_id: int):
        cams, _ = load_cameras_from_db()
        cam = next((c for c in cams if int(c.get("id", -1)) == int(cam_id)), None)
        if not cam or not cam.get("enabled", True):
            abort(404)
        data = _grab_one_jpeg(cam, timeout_s=6.0)
        if not data:
            abort(502)
        return Response(data, mimetype="image/jpeg")

    # ─────────────────────────────────────────────────────────────────────────────
    # Allow toggling active state for MULTIPLE cameras (no single-select limit)
    # ─────────────────────────────────────────────────────────────────────────────
    @app.route("/camera_active", methods=["POST"])
    @login_required
    def camera_active():
        try:
            cam_id = int(request.form.get("cam_id", "-1"))
            active = request.form.get("active", "1") in ("1", "true", "on", "yes")
            set_camera_active_state(cam_id, active)
            flash(f"Camera {cam_id} set_active={active}", "success")
        except Exception as e:
            logger.error(f"[CONTROL] set active failed: {e}")
            flash("Could not update camera active state.", "error")
        return redirect(request.referrer or url_for("control"))

    @app.route("/snapshots/<filename>")
    @login_required
    def serve_snapshot(filename):
        try:
            return send_from_directory(SNAPSHOT_FOLDER, filename)
        except FileNotFoundError:
            return "File not found", 404
        
    # --- Helpers for export hardening -------------------------------------------
    def _sanitize_export_record(row: dict, trim_len=64, noimages=False, max_image_bytes=1_500_000) -> dict:
        """
        Make every value safe for ReportLab. Optionally drop/limit images.
        - Truncates long strings to trim_len.
        - Converts lists/dicts/bytes to short markers.
        - If noimages=True or file too large/missing, clears snapshot_path.
        """
        safe = {}
        for k, v in (row or {}).items():
            try:
                if k == "snapshot_path":
                    sp = str(v or "")
                    if not sp or noimages:
                        safe[k] = ""
                    else:
                        try:
                            ok = os.path.exists(sp) and os.path.getsize(sp) <= max_image_bytes
                        except Exception:
                            ok = False
                        safe[k] = sp if ok else ""  # empty → report.py prints "N/A"
                    continue

                if v is None:
                    safe[k] = ""
                elif isinstance(v, (bytes, bytearray)):
                    safe[k] = (v.decode("utf-8", "ignore"))[:trim_len]
                elif isinstance(v, (list, tuple, set)):
                    safe[k] = f"[{len(v)} items]"
                elif isinstance(v, dict):
                    safe[k] = f"{{{len(v)} keys}}"
                else:
                    s = str(v)
                    safe[k] = s if len(s) <= trim_len else (s[:trim_len-1] + "…")
            except Exception:
                safe[k] = str(v)[:trim_len]
        return safe

    def _rows_to_export(cols_discovered, base_cols, optional_cols):
        select_cols = [c for c in base_cols if c in cols_discovered] + [c for c in optional_cols if c in cols_discovered]
        if not select_cols:
            select_cols = ["datetime","sensor","object_id","type","speed_kmh","distance","direction","snapshot_path","snapshot_type"]
        quoted = ", ".join(f'"{c}"' for c in select_cols)
        return quoted

    # --- helpers (place near your other small helpers) ---
    def _normalize_direction(d: str) -> str:
        d = (d or "").strip().lower()
        if d in {"approaching", "towards", "inbound", "forward"}: return "approaching"
        if d in {"departing", "away", "outbound", "backward"}:     return "departing"
        if d in {"stationary", "static", "stopped"}:               return "stationary"
        if d in {"right", "r"}:                                    return "right"
        if d in {"left", "l"}:                                     return "left"
        return "unknown"

    def _collect_summary(data):
        from collections import Counter
        speeds, types, dirs, snap_types = [], [], [], []
        for row in data:
            try:
                v = row.get("speed_kmh")
                if v is not None: speeds.append(float(v))
            except Exception:
                pass
            if row.get("type"):        types.append(str(row["type"]).upper())
            if row.get("direction"):   dirs.append(_normalize_direction(row.get("direction")))
            if row.get("snapshot_type"): snap_types.append(row["snapshot_type"])

        c = Counter(dirs)
        summary = {
            "total_records": len(data),
            "manual_snapshots": snap_types.count("manual"),
            "auto_snapshots":   snap_types.count("auto"),
            "avg_speed":   round(sum(speeds)/len(speeds), 2) if speeds else 0.0,
            "top_speed":   max(speeds) if speeds else 0.0,
            "lowest_speed": min(speeds) if speeds else 0.0,
            "most_detected_object": (Counter(types).most_common(1)[0][0] if types else "N/A"),
            # existing three (kept for compatibility)
            "approaching_count": c.get("approaching", 0),
            "stationary_count":  c.get("stationary", 0),
            "departing_count":   c.get("departing", 0),
            # new totals
            "right_count":   c.get("right", 0),
            "left_count":    c.get("left", 0),
            "unknown_count": c.get("unknown", 0),
            "last_detection": data[0].get("datetime") if data else "N/A",
            "speed_limits": load_config().get("dynamic_speed_limits", {})
        }
        return summary

    @app.route("/export")
    @login_required
    def export_csv():
        """
        CSV twin of /export_pdf:
        - If any filter params exist, delegate to /export_filtered_csv
        - Otherwise export a capped 'full' CSV (no filters)
        """
        try:
            params = request.args.to_dict()
            filter_keys = {
                "type","min_speed","max_speed","direction","object_id","selected",
                "snapshot_type","reviewed_only","flagged_only","unannotated_only",
                "min_confidence","max_confidence","start_date","end_date","motion_state",
                "max"
            }
            if any(k in params and str(params[k]).strip() != "" for k in filter_keys):
                # Delegate to filtered CSV exporter
                return export_filtered_csv()

            # ---- Full CSV (no filters) ----
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # discover available columns in radar_data, mirroring /export_pdf
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'radar_data'
                """)
                cols = {r[0] for r in cur.fetchall()}

            base_cols = [
                "datetime","sensor","object_id","type","confidence","speed_kmh","velocity","distance",
                "direction","motion_state","signal_level","doppler_frequency","reviewed","flagged",
                "snapshot_path","snapshot_type"
            ]
            optional_cols = [
                "range","azimuth","elevation","snr","velx","vely","velz","accx","accy","accz",
                "x","y","z","snapshot_status"
            ]
            if "plate_text" in cols:      optional_cols.append("plate_text")
            if "plate_conf" in cols:      optional_cols.append("plate_conf")
            if "plate_crop_path" in cols: optional_cols.append("plate_crop_path")
            select_cols = [c for c in base_cols if c in cols] + [c for c in optional_cols if c in cols]
            select_sql  = ", ".join(select_cols)

            cfg = load_config()
            max_rows = int(request.args.get("max") or cfg.get("export_max_rows", 200))
            max_rows = max(1, min(max_rows, 1000))

            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute(f"""
                    SELECT {select_sql}
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
                    ORDER BY measured_at DESC
                    LIMIT %s
                """, (max_rows,))
                rows = cur.fetchall()

            data = [dict(r) for r in rows]
            for row in data:
                if "datetime" in row and row["datetime"] is not None:
                    row["datetime"] = to_ist(row["datetime"])

            # Write CSV to /backups just like PDFs
            os.makedirs("backups", exist_ok=True)
            outfile = os.path.join("backups", f"radar_full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(outfile, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=select_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
            return send_file(outfile, as_attachment=True)

        except Exception as e:
            logger.exception("[EXPORT_CSV_ERROR]")
            return str(e), 500

    @app.route("/export_filtered_csv")
    @login_required
    def export_filtered_csv():
        """
        CSV version of /export_filtered_pdf — identical filters & hard cap.
        """
        try:
            cfg = load_config()
            HARD_CAP = int(cfg.get("export_hard_cap", 2000))
            requested_max = request.args.get("max")
            requested_max = int(requested_max) if requested_max else None
            params  = request.args.to_dict()

            # discover available columns (same as filtered PDF)
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'radar_data'
                """)
                cols = {r[0] for r in cur.fetchall()}

            base_cols = [
                "datetime", "sensor", "object_id", "type", "confidence",
                "speed_kmh", "velocity", "distance", "direction", "motion_state",
                "signal_level", "doppler_frequency", "reviewed", "flagged",
                "snapshot_path", "snapshot_type"
            ]
            optional_cols = ["range","azimuth","elevation","snr","velx","vely","velz",
                             "accx","accy","accz","x","y","z","snapshot_status"]
            if "plate_text" in cols:      optional_cols.append("plate_text")
            if "plate_conf" in cols:      optional_cols.append("plate_conf")
            if "plate_crop_path" in cols: optional_cols.append("plate_crop_path")
            select_cols = [c for c in base_cols if c in cols] + [c for c in optional_cols if c in cols]
            select_sql  = ", ".join(select_cols)

            basename_sql = "substring(snapshot_path from '[^/]+$')"
            where = [
                "snapshot_path IS NOT NULL",
                "snapshot_path <> ''",
                f"{basename_sql} NOT ILIKE %s"
            ]
            sqlp  = ['%\\_plate.jpg']

            # Selected basenames
            selected_raw = (params.get("selected") or "").strip()
            if selected_raw:
                selected_list = [os.path.basename(s) for s in selected_raw.split(",") if s.strip()]
                if selected_list:
                    ph = ", ".join(["%s"] * len(selected_list))
                    where.append(f"{basename_sql} IN ({ph})")
                    sqlp.extend(selected_list)

            # Type (LIKE)
            t = (params.get("type") or "").strip().upper()
            if t:
                where.append("UPPER(COALESCE(type,'')) LIKE %s")
                sqlp.append(f"%{t}%")

            # Speed bounds
            if params.get("min_speed"):
                try:
                    where.append("speed_kmh >= %s")
                    sqlp.append(float(params["min_speed"]))
                except ValueError:
                    pass
            if params.get("max_speed"):
                try:
                    where.append("speed_kmh <= %s")
                    sqlp.append(float(params["max_speed"]))
                except ValueError:
                    pass

            # Direction (exact, lower)
            d = (params.get("direction") or "").strip().lower()
            if d:
                where.append("LOWER(COALESCE(direction,'')) = %s")
                sqlp.append(d)

            # Motion state (exact, lower)
            ms = (params.get("motion_state") or "").strip().lower()
            if ms:
                where.append("LOWER(COALESCE(motion_state,'')) = %s")
                sqlp.append(ms)

            # Object ID (contains)
            obj_id = (params.get("object_id") or "").strip()
            if obj_id:
                where.append("CAST(object_id AS TEXT) ILIKE %s")
                sqlp.append(f"%{obj_id}%")

            # Snapshot type/manual/auto
            snap_type = (params.get("snapshot_type") or "").strip().lower()
            if snap_type in ("manual", "auto"):
                where.append("snapshot_type = %s")
                sqlp.append(snap_type)

            # Reviewed/Flagged toggles
            if params.get("reviewed_only") == "1":
                where.append("reviewed = 1")
            if params.get("flagged_only") == "1":
                where.append("flagged = 1")
            if params.get("unannotated_only") == "1":
                where.append("reviewed = 0 AND flagged = 0")

            # Confidence bounds
            if params.get("min_confidence"):
                try:
                    where.append("confidence >= %s")
                    sqlp.append(float(params["min_confidence"]))
                except ValueError:
                    pass
            if params.get("max_confidence"):
                try:
                    where.append("confidence <= %s")
                    sqlp.append(float(params["max_confidence"]))
                except ValueError:
                    pass

            # Date range (inclusive start, exclusive end+1d)
            start_date = (params.get("start_date") or "").strip()
            end_date   = (params.get("end_date")   or "").strip()
            if start_date:
                where.append("measured_at >= %s::date")
                sqlp.append(start_date)
            if end_date:
                where.append("measured_at < (%s::date + INTERVAL '1 day')")
                sqlp.append(end_date)

            where_sql = " AND ".join(where)

            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                # COUNT for cap parity with filtered PDF
                cur.execute(f"SELECT COUNT(*) AS c FROM radar_data WHERE {where_sql}", sqlp)
                filtered_total = int(cur.fetchone()["c"])
                limit = requested_max if requested_max else min(filtered_total, HARD_CAP)
                limit = max(1, min(limit, HARD_CAP))

                cur.execute(f"""
                    SELECT {select_sql}
                    FROM radar_data
                    WHERE {where_sql}
                    ORDER BY measured_at DESC
                    LIMIT %s
                """, sqlp + [limit])
                rows = cur.fetchall()

            data = [dict(r) for r in rows]
            for row in data:
                if "datetime" in row and row["datetime"] is not None:
                    row["datetime"] = to_ist(row["datetime"])

            os.makedirs("backups", exist_ok=True)
            outfile = os.path.join("backups", f"radar_filtered_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(outfile, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=select_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
            return send_file(outfile, as_attachment=True)

        except Exception as e:
            logger.exception("[EXPORT_FILTERED_CSV_ERROR]")
            return str(e), 500

    @app.route("/export_pdf")
    @login_required
    def export_pdf():
        try:
            # If any filter params are present, delegate to the filtered exporter
            params = request.args.to_dict()
            filter_keys = {
                "type","min_speed","max_speed","direction","object_id","selected",
                "snapshot_type","reviewed_only","flagged_only","unannotated_only",
                "max","nocharts","noimages"
            }
            if any(k in params and str(params[k]).strip() != "" for k in filter_keys):
                return export_filtered_pdf()

            # ---- Full export (no filters) ----
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # discover available columns in radar_data
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'radar_data'
                """)
                cols = {r[0] for r in cur.fetchall()}

            base_cols = [
                "datetime","sensor","object_id","type","confidence","speed_kmh","velocity","distance",
                "direction","motion_state","signal_level","doppler_frequency","reviewed","flagged",
                "snapshot_path","snapshot_type"
            ]
            optional_cols = [
                "range","azimuth","elevation","snr","velx","vely","velz","accx","accy","accz",
                "x","y","z","snapshot_status"
            ]
            if "plate_text" in cols:       optional_cols.append("plate_text")
            if "plate_conf" in cols:       optional_cols.append("plate_conf")
            if "plate_crop_path" in cols:  optional_cols.append("plate_crop_path")
            select_cols = [c for c in base_cols if c in cols] + [c for c in optional_cols if c in cols]
            select_sql = ", ".join(select_cols)

            # Allow caller to cap rows to avoid OOM on Pi
            cfg = load_config()
            max_rows = int(request.args.get("max") or cfg.get("export_max_rows", 200))
            max_rows = max(1, min(max_rows, 1000))

            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute(f"""
                    SELECT {select_sql}
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
                    ORDER BY measured_at DESC
                    LIMIT %s
                """, (max_rows,))
                rows = cur.fetchall()

            data = [dict(r) for r in rows]
            # Render all datetimes as IST strings for the report
            for row in data:
                if "datetime" in row:
                    row["datetime"] = to_ist(row["datetime"])

            # Build summary with direction counts, etc.
            from collections import Counter
            speeds, types, directions, snap_types = [], [], [], []
            for row in data:
                v = row.get("speed_kmh")
                if v is not None:
                    try: speeds.append(float(v))
                    except: pass
                if row.get("type"):          types.append(str(row["type"]).upper())
                if row.get("direction"):     directions.append(str(row["direction"]).lower())
                if row.get("snapshot_type"): snap_types.append(row["snapshot_type"])

            summary = _collect_summary(data)

            # (Optional) charts
            charts = {}
            try:
                resp = requests.get("http://127.0.0.1:5000/api/charts?days=0", timeout=2.5)
                if resp.ok:
                    charts = resp.json()
            except Exception as e:
                logger.warning(f"[CHART FETCH] {e}")

            logo_path = "C:\ESSI\Projects\isk6843\static\essi_logo.jpeg"
            filename  = f"radar_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath  = os.path.join("backups", filename)
            os.makedirs("backups", exist_ok=True)

            from report import generate_pdf_report
            generate_pdf_report(filepath, data=data, summary=summary, logo_path=logo_path, charts=charts)
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype="application/pdf"
            )

        except Exception as e:
            logger.exception("[EXPORT_PDF_ERROR]")
            return str(e), 500

    @app.route("/export_filtered_pdf")
    @login_required
    def export_filtered_pdf():
        try:
            cfg = load_config()
            HARD_CAP = int(cfg.get("export_hard_cap", 2000))
            requested_max = request.args.get("max")
            requested_max = int(requested_max) if requested_max else None
            params  = request.args.to_dict()
            filters = params.copy()  # passed to the PDF so it lists them

            # discover columns (unchanged)
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'radar_data'
                """)
                cols = {r[0] for r in cur.fetchall()}

            base_cols = [
                "datetime", "sensor", "object_id", "type", "confidence",
                "speed_kmh", "velocity", "distance", "direction", "motion_state",
                "signal_level", "doppler_frequency", "reviewed", "flagged",
                "snapshot_path"
            ]
            optional_cols = ["range","azimuth","elevation","snr","velx","vely","velz",
                            "accx","accy","accz","x","y","z","snapshot_status"]
            if "plate_text" in cols:       optional_cols.append("plate_text")
            if "plate_conf" in cols:       optional_cols.append("plate_conf")
            if "plate_crop_path" in cols:  optional_cols.append("plate_crop_path")
            select_cols = [c for c in base_cols if c in cols] + [c for c in optional_cols if c in cols]
            select_sql  = ", ".join(select_cols)

            # --- build WHERE + params (match /gallery)
            basename_sql = "substring(snapshot_path from '[^/]+$')"
            where = [
                "snapshot_path IS NOT NULL",
                "snapshot_path <> ''",
                f"{basename_sql} NOT ILIKE %s"
            ]
            sqlp  = ['%\\_plate.jpg']

            # Selected basenames (comma-separated) — intersects with other filters
            selected_raw = (params.get("selected") or "").strip()
            if selected_raw:
                selected_list = [os.path.basename(s) for s in selected_raw.split(",") if s.strip()]
                if selected_list:
                    ph = ", ".join(["%s"] * len(selected_list))
                    where.append(f"{basename_sql} IN ({ph})")
                    sqlp.extend(selected_list)

            t = (params.get("type") or "").strip().upper()
            if t:
                where.append("UPPER(COALESCE(type,'')) LIKE %s")
                sqlp.append(f"%{t}%")

            if params.get("min_speed") or params.get("max_speed"):
                min_speed = float(params.get("min_speed") or 0)
                max_speed = float(params.get("max_speed") or 999)
                where.append("speed_kmh BETWEEN %s AND %s")
                sqlp += [min_speed, max_speed]

            d = (params.get("direction") or "").strip().lower()
            if d:
                where.append("LOWER(COALESCE(direction,'')) LIKE %s")
                sqlp.append(f"%{d}%")

            motion_state = (params.get("motion_state") or "").strip().lower()
            if motion_state:
                where.append("LOWER(COALESCE(motion_state,'')) LIKE %s")
                sqlp.append(f"%{motion_state}%")

            oid = (params.get("object_id") or "").strip()
            if oid:
                where.append("COALESCE(object_id,'') LIKE %s")
                sqlp.append(f"%{oid}%")

            start_date = (params.get("start_date") or "").strip()
            if start_date:
                where.append("DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) >= %s")
                sqlp.append(start_date)

            end_date = (params.get("end_date") or "").strip()
            if end_date:
                where.append("DATE(COALESCE(datetime::TIMESTAMP, to_timestamp(timestamp))) <= %s")
                sqlp.append(end_date)

            if params.get("min_confidence") or params.get("max_confidence"):
                min_conf = float(params.get("min_confidence") or 0)
                max_conf = float(params.get("max_confidence") or 1)
                where.append("confidence BETWEEN %s AND %s")
                sqlp += [min_conf, max_conf]

            snap_type = (params.get("snapshot_type") or "").strip().lower()
            if snap_type in ("manual", "auto"):
                where.append("snapshot_type = %s")
                sqlp.append(snap_type)

            if params.get("reviewed_only") == "1":
                where.append("reviewed = 1")
            if params.get("flagged_only") == "1":
                where.append("flagged = 1")
            if params.get("unannotated_only") == "1":
                where.append("reviewed = 0 AND flagged = 0")

            where_sql = " AND ".join(where)

            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # COUNT first so our default LIMIT equals the filtered size
                cur.execute(f"SELECT COUNT(*) AS c FROM radar_data WHERE {where_sql}", sqlp)
                filtered_total = int(cur.fetchone()["c"])
                limit = requested_max if requested_max else min(filtered_total, HARD_CAP)
                limit = max(1, min(limit, HARD_CAP))

                cur.execute(f"""
                    SELECT {select_sql}
                    FROM radar_data
                    WHERE {where_sql}
                    ORDER BY measured_at DESC
                    LIMIT %s
                """, sqlp + [limit])
                rows = cur.fetchall()

            data = [dict(r) for r in rows]
            # Render all datetimes as IST strings for the report
            for row in data:
                if "datetime" in row:
                    row["datetime"] = to_ist(row["datetime"])

            # summary identical to /export_pdf
            from collections import Counter
            speeds, types, directions, snap_types = [], [], [], []
            for row in data:
                v = row.get("speed_kmh")
                if v is not None:
                    try: speeds.append(float(v))
                    except: pass
                if row.get("type"):          types.append(str(row["type"]).upper())
                if row.get("direction"):     directions.append(str(row["direction"]).lower())
                if row.get("snapshot_type"): snap_types.append(row["snapshot_type"])

            summary = _collect_summary(data)

            charts = {}
            try:
                resp = requests.get("http://127.0.0.1:5000/api/charts?days=0", timeout=2.5)
                if resp.ok:
                    charts = resp.json()
            except Exception as e:
                logger.warning(f"[CHART FETCH] {e}")

            logo_path = "C:\ESSI\Projects\isk6843\static\essi_logo.jpeg"
            filename  = f"radar_filtered_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath  = os.path.join("backups", filename)
            os.makedirs("backups", exist_ok=True)

            from report import generate_pdf_report
            generate_pdf_report(
                filepath,
                title="Radar Based Speed Detection Report — Filtered",
                summary=summary,
                data=data,
                filters=filters,
                logo_path=logo_path,
                charts=charts
            )
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype="application/pdf"
            )
        except Exception as e:
            logger.error(f"[EXPORT_FILTERED_PDF_ERROR] {e}")
            return str(e), 500

    @app.route("/snapshot_pdf/<path:filename>")
    @login_required
    def snapshot_pdf(filename):
        try:
            with get_db_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                # Match by basename (robust) and fetch plate fields same as full/filtered
                basename_sql = "substring(snapshot_path from '[^/]+$')"
                cur.execute(f"""
                    SELECT datetime, sensor, object_id, type, confidence, speed_kmh, velocity,
                           distance, direction, motion_state,
                           signal_level, doppler_frequency, reviewed, flagged,
                           snapshot_path, snapshot_type,
                           clip_path, clip_status,
                           plate_text, plate_conf, plate_crop_path
                    FROM radar_data
                    WHERE snapshot_path IS NOT NULL AND snapshot_path <> ''
                      AND {basename_sql} = %s
                    ORDER BY (plate_text IS NOT NULL AND plate_text <> '') DESC,
                             measured_at DESC
                    LIMIT 1
                """, (filename,))
                row = cur.fetchone()

            if not row:
                abort(404, description="Snapshot not found")

            rec = dict(row)
            rec["radar_distance"]  = float(rec.get("distance") or 0.0)

            img_fp = rec.get("snapshot_path")
            if not img_fp or not os.path.exists(img_fp):
                abort(404, description="Image file missing")

            # Generate single-detection PDF to a temp path
            logo_path = "C:\ESSI\Projects\isk6843\static\essi_logo.jpeg"
            out_dir   = "backups"
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(filename))[0]
            pdf_fp = os.path.join(out_dir, f"{base}_detection.pdf")
            from report import generate_single_detection_pdf
            generate_single_detection_pdf(pdf_fp, record=rec, logo_path=logo_path)

            # Build ZIP bundle in-memory (store only to minimize CPU)
            buf = BytesIO()
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                # PDF
                if os.path.exists(pdf_fp):
                    z.write(pdf_fp, arcname=os.path.basename(pdf_fp))
                # Image
                z.write(img_fp, arcname=os.path.basename(img_fp))
                # Clip (if ready & present)
                clip_fp = (rec.get("clip_path") or "").strip()
                clip_ok = clip_fp and os.path.exists(clip_fp) and str(rec.get("clip_status") or "").lower() == "ready"
                if clip_ok:
                    z.write(clip_fp, arcname=os.path.basename(clip_fp))
            buf.seek(0)
            download_name = f"{base}_bundle.zip"
            return send_file(buf, as_attachment=True, download_name=download_name, mimetype="application/zip")

        except Exception as e:
            logger.exception("[SNAPSHOT_PDF] error")
            return str(e), 500

    @app.route("/download_detection_bundle/<path:filename>")
    @login_required
    def download_detection_bundle(filename):
        try:
            # 1) Look up the DB row by snapshot basename
            with get_db_connection() as conn:
                rec = _row_by_snapshot_basename(conn, filename)
            if not rec:
                return ("Snapshot not found", 404)

            snap_path = rec.get("snapshot_path")
            if not snap_path or not os.path.exists(snap_path):
                return ("Image file missing", 404)

            base = os.path.splitext(os.path.basename(filename))[0]

            # 2) Generate the single-detection PDF (temp) using existing report code
            logo_path = "C:\ESSI\Projects\isk6843\static\essi_logo.jpeg"
            os.makedirs(BACKUP_FOLDER, exist_ok=True)
            pdf_tmp = os.path.join(BACKUP_FOLDER, f"{base}_detection.pdf")
            try:
                from report import generate_single_detection_pdf
                # report expects 'distance' etc; pass record as-is
                generate_single_detection_pdf(pdf_tmp, record=dict(rec), logo_path=logo_path)
                pdf_ok = os.path.exists(pdf_tmp)
            except Exception as e:
                logger.error(f"[DL BUNDLE] PDF generation failed: {e}")
                pdf_ok = False

            # 3) Prefer items from the violation bundle if available
            bundle_dir = _guess_bundle_from_snapshot_path(snap_path)
            img_src  = snap_path
            meta_src = None
            clip_src = None
            extra_files = []
            if bundle_dir and os.path.isdir(bundle_dir):
                b_img  = os.path.join(bundle_dir, "image.jpg")
                b_meta = os.path.join(bundle_dir, "meta.json")
                b_clip = os.path.join(bundle_dir, "clip.mp4")
                if os.path.exists(b_img):  img_src = b_img
                if os.path.exists(b_meta): meta_src = b_meta
                if os.path.exists(b_clip): clip_src = b_clip
                for root, _, files in os.walk(bundle_dir):
                    for f in files:
                        lf = f.lower()
                        if lf.endswith((".jpg", ".jpeg", ".png", ".mp4", ".mov")) and f not in {
                            os.path.basename(b_img) if os.path.exists(b_img) else "",
                            os.path.basename(b_meta) if os.path.exists(b_meta) else "",
                            os.path.basename(b_clip) if os.path.exists(b_clip) else "",
                        }:
                            extra_files.append(os.path.join(root, f))
            # 4) Fallbacks: synthesize metadata; locate clip if DB says ready
            cleanup_paths = []
            if not meta_src:
                try:
                    # make a JSON that's safe to share
                    safe = {k: (float(v) if isinstance(v, (int, float)) else (str(v) if v is not None else None))
                            for k, v in dict(rec).items()}
                    meta_tmp = os.path.join(BACKUP_FOLDER, f"{base}_meta.json")
                    with open(meta_tmp, "w") as f:
                        json.dump(safe, f, indent=2)
                    meta_src = meta_tmp
                    cleanup_paths.append(meta_tmp)
                except Exception as e:
                    logger.error(f"[DL BUNDLE] meta synth failed: {e}")
                    meta_src = None

            if not clip_src:
                clip_fp = (rec.get("clip_path") or "").strip()
                clip_ok = clip_fp and os.path.exists(clip_fp) and str(rec.get("clip_status") or "").lower() == "ready"
                if clip_ok:
                    clip_src = clip_fp

            # 5) Build ZIP in-memory
            buf = BytesIO()
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                if pdf_ok:
                    z.write(pdf_tmp, arcname=os.path.basename(pdf_tmp))
                    cleanup_paths.append(pdf_tmp)  # clean after response
                if img_src and os.path.exists(img_src):
                    z.write(img_src, arcname=os.path.basename(img_src))
                if meta_src and os.path.exists(meta_src):
                    # If using bundle meta.json, preserve its name
                    z.write(meta_src, arcname=os.path.basename(meta_src))
                if clip_src and os.path.exists(clip_src):
                    z.write(clip_src, arcname=os.path.basename(clip_src))
                try:
                    if bundle_dir:
                        sp = os.path.join(bundle_dir, "seal.json")
                        if os.path.exists(sp):
                            z.write(sp, arcname="seal.json")
                except Exception:
                    pass
                for ef in extra_files:
                    try:
                        z.write(ef, arcname=os.path.basename(ef))
                    except Exception as e:
                        logger.warning(f"[DL BUNDLE] skip extra {ef}: {e}")
            buf.seek(0)

            @after_this_request
            def _cleanup(response):
                for p in cleanup_paths:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                return response

            return send_file(
                buf,
                as_attachment=True,
                download_name=f"{base}_bundle.zip",
                mimetype="application/zip"
            )

        except Exception as e:
            logger.exception("[DOWNLOAD_DETECTION_BUNDLE] error")
            return (str(e), 500)

    @app.route("/bundle/open")
    @login_required
    def bundle_open():
        """
        Minimal HTML listing (no new template required).
        Query: ?snapshot=<basename.jpg>
        """
        snap = (request.args.get("snapshot") or "").strip()
        if not snap:
            return "snapshot param required", 400
        bdir = _resolve_bundle_by_basename(snap)
        if not bdir or not os.path.isdir(bdir):
            return f"No bundle found for {snap}", 404
        man = _bundle_manifest(bdir)
        # link to existing endpoints (snapshots/, /clips/<file>)
        img_url  = url_for("serve_snapshot", filename=snap)
        clip_url = None
        if man.get("clip"):
            # the clip in bundle is 'clip.mp4', but public URL is via /clips/<original>.mp4 symlink.
            # If we cannot guess original, serve from bundle via a one-off file sender:
            try:
                with get_db_connection() as conn:
                    row = _row_by_snapshot_basename(conn, snap)
                if row and row.get("clip_path"):
                    clip_url = url_for("serve_clip", filename=os.path.basename(row["clip_path"]))
            except Exception:
                clip_url = None
        zip_url = url_for("bundle_zip", snapshot=snap)
        html = f"""
        <html><head><meta charset="utf-8"><title>Bundle — {snap}</title>
        <style>body{{font-family:system-ui,Segoe UI,Arial;margin:16px}} .row{{margin:6px 0}}</style></head>
        <body>
          <h2>Bundle for: {snap}</h2>
          <div class="row"><b>Folder:</b> {man.get('dir')}</div>
          <div class="row"><b>Image:</b> {'present' if man.get('image') else 'missing'} &nbsp; 
              {'<a href="'+img_url+'" target="_blank">open image</a>' if man.get('image') else ''}</div>
          <div class="row"><b>Clip:</b> {'present' if man.get('clip') else 'missing'} &nbsp;
              {(('<a href="'+clip_url+'" target="_blank">play clip</a>') if clip_url else '')}</div>
          <div class="row"><b>Meta:</b> {'present' if man.get('meta') else 'missing'}</div>
          <div class="row"><a href="{zip_url}">Download bundle (.zip)</a></div>
        </body></html>
        """.strip()
        return Response(html, mimetype="text/html")

    # ────────────────────────────────────────────────────────────────────────────
    # Evidence Seal verifier (PUBLIC)
    # ────────────────────────────────────────────────────────────────────────────
    def _find_bundle_by_seal_id(seal_id: str) -> str | None:
        """Linear scan for seal.json with matching payload.seal_id under violations/."""
        try:
            root = BUNDLES_DIR
            if not os.path.isdir(root):
                return None
            for day in os.listdir(root):
                day_dir = os.path.join(root, day)
                if not os.path.isdir(day_dir):
                    continue
                for name in os.listdir(day_dir):
                    bdir = os.path.join(day_dir, name)
                    sp = os.path.join(bdir, "seal.json")
                    if not os.path.isfile(sp):
                        continue
                    try:
                        data = json.load(open(sp, "r"))
                        payload = data.get("payload") or {}
                        if (payload.get("seal_id") or "").strip() == seal_id:
                            return bdir
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    @app.route("/verify/<seal_id>")
    def verify_evidence(seal_id: str):
        """
        Public verifier: recompute hashes and Ed25519-signature for a bundle's seal.json.
        Returns tiny HTML (✓/✗) or JSON when Accept: application/json or ?json=1.
        """
        try:
            bdir = _find_bundle_by_seal_id(seal_id.strip())
            if not bdir:
                msg = {"status": "not_found", "error": "Seal not found"}
                wants_json = ("application/json" in (request.headers.get("Accept") or "")) or request.args.get("json") == "1"
                return (jsonify(msg), 404) if wants_json else (f"<h2>✗ Not found</h2><div>seal_id={seal_id}</div>", 404)

            # Verify on disk
            from evidence_seal import verify_seal
            ok, details, reason = verify_seal(bdir)

            payload = details.get("payload", {}) or {}
            created_at_raw = payload.get("created_at")
            # Parse ISO string (accept trailing 'Z'), then format via to_ist(dt)
            try:
                _dt = datetime.fromisoformat(str(created_at_raw).replace("Z", "+00:00")) if created_at_raw else None
            except Exception:
                _dt = None
            created_at_ist = to_ist(_dt) if _dt else "—"
            server_time_ist = to_ist(datetime.now(timezone.utc))

            # Useful URLs using existing endpoints
            base = os.path.basename(bdir) + ".jpg"  # matches snapshots symlink name
            img_url = url_for("serve_snapshot", filename=base)
            zip_url = url_for("bundle_zip", snapshot=base)

            wants_json = ("application/json" in (request.headers.get("Accept") or "").lower()) or request.args.get("json") == "1"
            verdict = "verified" if ok else "tampered"
            if wants_json:
                return jsonify({
                    "status": verdict,
                    "seal_id": payload.get("seal_id"),
                    "created_at": created_at_raw,
                    "created_at_ist": created_at_ist,
                    "server_time_ist": server_time_ist,
                    "site": payload.get("site"),
                    "files_expected": (details.get("payload", {}) or {}).get("files") or {},
                    "files_recomputed": details.get("recomputed") or {},
                    "sig_ok": bool(details.get("sig_ok")),
                    "reason": (None if ok else reason),
                    "image_url": img_url,
                    "download_zip": zip_url,
                }), 200

            # HTML
            mark = "✓ Verified" if ok else "✗ Tampered"
            color = "#1a7f37" if ok else "#c62828"
            files = payload.get("files", {}) or {}
            recomputed = details.get("recomputed", {}) or {}
            rows = []
            for k in ("image.jpg", "meta.json", "clip.mp4"):
                exp = files.get(k)
                got = recomputed.get(k)
                ok_f = (exp == got)
                rows.append(f"""
                  <tr>
                    <td style="padding:6px 10px">{k}</td>
                    <td style="padding:6px 10px; font-family:monospace">{(exp or '—')}</td>
                    <td style="padding:6px 10px; font-family:monospace">{(got or '—')}</td>
                    <td style="padding:6px 10px">{'✓' if ok_f else '✗'}</td>
                  </tr>
                """)

            reason_html = "" if ok else f'<div class="row"><b>Reason:</b> {reason}</div>'
            html = f"""
              <html>
              <head><meta charset="utf-8"><title>Verify — {seal_id}</title></head>
              <body style="font-family:system-ui,Segoe UI,Arial;margin:16px">
                <h2 style="color:{color}">{mark}</h2>
                <div class="row"><b>Seal ID:</b> {payload.get('seal_id') or seal_id}</div>
                <div class="row"><b>Created at (IST):</b> {created_at_ist}</div>
                <div class="row"><b>Server time (IST):</b> {server_time_ist}</div>
                <div class="row"><b>Site:</b> {payload.get('site') or '—'}</div>
                {reason_html}
                <div class="row" style="margin-top:10px">
                  <a href="{img_url}" target="_blank">Open image</a> ·
                  <a href="{zip_url}">Download bundle (.zip)</a>
                </div>
                <h3>Files</h3>
                <table border="1" cellpadding="0" cellspacing="0" style="border-collapse:collapse">
                  <thead>
                    <tr>
                      <th style="padding:6px 10px;text-align:left">File</th>
                      <th style="padding:6px 10px;text-align:left">Expected SHA-256</th>
                      <th style="padding:6px 10px;text-align:left">Recomputed SHA-256</th>
                      <th style="padding:6px 10px;text-align:left">OK</th>
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(rows)}
                  </tbody>
                </table>
              </body>
              </html>
            """.strip()
            return Response(html, mimetype="text/html")
        except Exception as e:
            logger.exception("[VERIFY] error")
            return ("Internal error", 500)

    @app.route("/api/bundle_upload", methods=["POST"])
    @login_required
    def api_bundle_upload():
        """
        Zip the entire violation bundle (all media) and upload to Drive via rclone.
        Body JSON: { "snapshot": "<basename.jpg>" }
        Returns a shareable link if the remote supports it.
        """
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            data = request.get_json(silent=True) or {}
            snap = (data.get("snapshot") or "").strip()
            if not snap:
                return jsonify({"error": "snapshot required"}), 400
            bdir = _resolve_bundle_by_basename(snap)
            if not (bdir and os.path.isdir(bdir)):
                return jsonify({"error": "bundle not found"}), 404
            base = os.path.splitext(snap)[0]
            os.makedirs(BACKUP_FOLDER, exist_ok=True)
            zip_fp = os.path.join(BACKUP_FOLDER, f"{base}_bundle.zip")
            # Build a fresh ZIP with all media
            with zipfile.ZipFile(zip_fp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                for root, _, files in os.walk(bdir):
                    for f in files:
                        src = os.path.join(root, f)
                        try:
                            z.write(src, arcname=os.path.basename(src))
                        except Exception as e:
                            logger.warning(f"[BUNDLE UPLOAD] skip {src}: {e}")
            # Upload ZIP
            link = _rclone_copy_and_link(zip_fp)
            return jsonify({"status": "ok", "gdrive_link": link, "zip_size": os.path.getsize(zip_fp)})
        except Exception as e:
            logger.exception("[BUNDLE UPLOAD] error")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    @app.route("/bundle/zip")
    @login_required
    def bundle_zip():
        """
        Download the entire bundle as a ZIP.
        Query: ?snapshot=<basename.jpg>
        """
        snap = (request.args.get("snapshot") or "").strip()
        if not snap:
            return "snapshot param required", 400
        bdir = _resolve_bundle_by_basename(snap)
        if not bdir or not os.path.isdir(bdir):
            return f"No bundle found for {snap}", 404
        # Build ZIP on-the-fly
        buf = BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for name in ("image.jpg", "clip.mp4", "meta.json", "seal.json"):
                p = os.path.join(bdir, name)
                try:
                    if os.path.exists(p):
                        z.write(p, arcname=name)
                except Exception:
                    pass
        buf.seek(0)
        base = os.path.splitext(snap)[0]
        return send_file(buf, as_attachment=True, download_name=f"{base}_bundle.zip", mimetype="application/zip")

    @app.route("/calibration/start", methods=["POST"])
    @login_required
    def calibration_start():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        cfg = load_config()
        cfg["calibration_mode"] = True
        save_config(cfg, write_reload_flag=True)
        cam_id = (request.get_json(silent=True) or {}).get("camera_id") or request.args.get("camera_id")
        if hasattr(reset_pairs, "__code__") and ("camera_id" in reset_pairs.__code__.co_varnames) and cam_id is not None:
            reset_pairs(camera_id=int(cam_id))
        else:
            reset_pairs()
        return jsonify({"status": "ok", "message": "Calibration mode ON, pairs cleared"})

    @app.route("/calibration/stop", methods=["POST"])
    @login_required
    def calibration_stop():
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        cfg = load_config()
        cfg["calibration_mode"] = False
        save_config(cfg, write_reload_flag=True)
        return jsonify({"status": "ok", "message": "Calibration mode OFF"})

    @app.route("/calibration/reset_pairs", methods=["POST"])
    @login_required
    def calibration_reset_pairs():
        """Clear collected (u,v) ↔ (r,az,el) pairs without toggling mode."""
        if not is_admin():
            return jsonify({"status": "error", "error": "Unauthorized"}), 403
        try:
            cam_id = (request.get_json(silent=True) or {}).get("camera_id") or request.args.get("camera_id")
            if hasattr(reset_pairs, "__code__") and ("camera_id" in reset_pairs.__code__.co_varnames) and cam_id is not None:
                reset_pairs(camera_id=int(cam_id))
            else:
                reset_pairs()
            return jsonify({"status": "ok", "message": "Cleared collected calibration pairs"})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/calibration/restart", methods=["POST"])
    @login_required
    def calibration_restart():
        """Delete any previous model(s), clear pairs, and start calibration."""
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        cam_id = (request.get_json(silent=True) or {}).get("camera_id") or request.args.get("camera_id")
        try:
            from calibration import _paths_for_camera as _paths_for_camera_fn
        except Exception:
            _paths_for_camera_fn = None
        if _paths_for_camera_fn and cam_id is not None:
            _, live_path, staged_path, _ = _paths_for_camera_fn(int(cam_id))
        else:
            _, live_path, staged_path, _ = _calib_paths()
        deleted = []
        if os.path.exists(live_path):
            os.remove(live_path); deleted.append("live")
        if os.path.exists(staged_path):
            os.remove(staged_path); deleted.append("staged")
        if hasattr(reset_pairs, "__code__") and ("camera_id" in reset_pairs.__code__.co_varnames) and cam_id is not None:
            reset_pairs(camera_id=int(cam_id))
        else:
            reset_pairs()
        cfg = load_config()
        cfg["calibration_mode"] = True
        save_config(cfg, write_reload_flag=True)
        return jsonify({"status": "ok", "deleted": deleted, "message": "Restarted calibration"})

    @app.route("/calibration/delete", methods=["POST"])
    @login_required
    def calibration_delete():
        """Delete previous model(s) (live/staged)."""
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        body = (request.json or {})
        which = body.get("which", "both")  # "live" | "staged" | "both"
        cam_id = body.get("camera_id") or request.args.get("camera_id")
        try:
            from calibration import _paths_for_camera as _paths_for_camera_fn
        except Exception:
            _paths_for_camera_fn = None
        if _paths_for_camera_fn and cam_id is not None:
            _, live_path, staged_path, _ = _paths_for_camera_fn(int(cam_id))
        else:
            _, live_path, staged_path, _ = _calib_paths()
        deleted = []
        try:
            if which in ("both", "live") and os.path.exists(live_path):
                os.remove(live_path); deleted.append("live")
            if which in ("both", "staged") and os.path.exists(staged_path):
                os.remove(staged_path); deleted.append("staged")
            return jsonify({"status": "ok", "deleted": deleted})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/calibration/fit_publish", methods=["POST"])
    @login_required
    def calibration_fit_publish():
        """Fit extrinsics using current pairs and publish live model in one shot."""
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        cfg = load_config()
        intr = estimate_intrinsics(
            int(cfg.get("camera_width_px", 640)),
            int(cfg.get("camera_height_px", 480)),
            float(cfg.get("camera_fov_h_deg", 90.0)),
            tuple(cfg.get("distortion", [])) if "distortion" in cfg else None
        )
        body = (request.get_json(silent=True) or {})
        cam_id = body.get("camera_id") or request.args.get("camera_id")
        _kwargs = {"min_points": int(cfg.get("calibration_min_points", 12))}
        if hasattr(fit_extrinsics, "__code__") and ("camera_id" in fit_extrinsics.__code__.co_varnames) and cam_id is not None:
            _kwargs["camera_id"] = int(cam_id)
        model, med_err, n_pairs = fit_extrinsics(intr, **_kwargs)
        if hasattr(publish_model, "__code__") and ("camera_id" in publish_model.__code__.co_varnames) and cam_id is not None:
            versioned = publish_model(camera_id=int(cam_id))
        else:
            versioned = publish_model()
        return jsonify({
            "status": "published",
            "pairs": n_pairs,
            "median_px": round(med_err, 2),
            "path": versioned
        })

    @app.route("/calibration/run", methods=["GET"])
    @login_required
    def calibration_run_compat():
        """Compat route from older links; just bounce to /control with wizard=1."""
        resp = redirect(url_for("control", wizard=1), code=302)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/calibration/wizard", methods=["GET"])
    @login_required
    def calibration_wizard():
        """Alias; serve wizard inside the control page."""
        resp = redirect(url_for("control", wizard=1), code=302)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/calibration/capture_shot", methods=["POST"])
    @login_required
    def api_calibration_capture_shot():
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            cfg = load_config()
            cams = cfg.get("cameras") or []
            body = (request.get_json(silent=True) or {})
            # allow explicit camera selection; fallback to current global
            raw_sel = body.get("camera_id", cfg.get("selected_camera", 0))
            cam = {}
            sel_id = None
            try:
                sel_id = int(raw_sel)
            except Exception:
                sel_id = None

            # Prefer matching by DB camera ID (robust across ordering),
            # then fall back to treating the value as a list index.
            if isinstance(cams, list) and cams:
                if sel_id is not None:
                    # Try ID match
                    for c in cams:
                        try:
                            if int(c.get("id")) == sel_id:
                                cam = c
                                break
                        except Exception:
                            pass
                if not cam:
                    # Fall back to index semantics
                    try:
                        idx = int(raw_sel)
                        if 0 <= idx < len(cams):
                            cam = cams[idx]
                    except Exception:
                        pass
            elif isinstance(cams, dict):
                cam = cams

            # Try snapshot_url first, then fall back to url. Log which one we used.
            primary_url = cam.get("snapshot_url") or cam.get("url")
            fallback_url = cam.get("url") if cam.get("snapshot_url") else None
            last_err = None
            snap_path = None
            for attempt_url in [u for u in [primary_url, fallback_url] if u]:
                try:
                    snap_path = capture_snapshot(
                        camera_url=attempt_url,
                        username=cam.get("username"),
                        password=cam.get("password")
                    )
                    if snap_path and os.path.exists(snap_path):
                        break
                except Exception as e:
                    last_err = str(e)
                    snap_path = None
            if not snap_path or not os.path.exists(snap_path):
                # Surface a helpful hint to the UI
                return jsonify({
                    "ok": False,
                    "error": "snapshot failed",
                    "details": last_err or "no frame from camera",
                    "camera_url": primary_url
                }), 500

            # Poll briefly (up to ~0.7s) to allow main loop to refresh between camera trigger and read
            candidates = []
            for _ in range(20):
                candidates = _load_latest_candidates(max_age_s=3.0)
                if candidates:
                    break
                time.sleep(0.1)

            base = os.path.basename(snap_path)
            if not base:
                return jsonify({"ok": False, "error": "bad snapshot path"}), 500
            snapshot_url = url_for("serve_snapshot", filename=base)

            # Persist sidecar next to snapshot for reproducibility
            try:
                with open(os.path.join(SNAPSHOT_FOLDER, base) + ".live.json", "w") as f:
                    json.dump({
                        "snapshot_path": os.path.join(SNAPSHOT_FOLDER, base),
                        "candidates": candidates,
                        "camera_id": sel_id if sel_id is not None else raw_sel
                    }, f)
            except Exception:
                pass

            return jsonify({
                "ok": True,
                "snapshot_path": os.path.join(SNAPSHOT_FOLDER, base),
                "snapshot_basename": base,
                "snapshot_url": snapshot_url,
                "candidates": candidates,
                "camera_id": sel_id if sel_id is not None else raw_sel
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/live_json_candidates")
    @login_required
    def api_calibration_live_json_candidates():
        """Debug helper: list all candidate live_objects.json paths with freshness info."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        now = time.time()
        report = []
        for p in _live_json_paths():
            try:
                exists = os.path.exists(p)
                size = os.path.getsize(p) if exists else None
                mtime = os.path.getmtime(p) if exists else None
                age = (now - mtime) if mtime else None
                head = ""
                if exists:
                    with open(p, "r") as f:
                        head = f.read(200)
                report.append({
                    "path": p, "exists": exists, "size": size,
                    "mtime": mtime, "age_s": age, "head_preview": head
                })
            except Exception as e:
                report.append({"path": p, "error": str(e)})
        return jsonify({"ok": True, "candidates": report})

    @app.route("/api/calibration/add_click", methods=["POST"])
    @login_required
    def api_calibration_add_click():
        """Add one (u,v)↔(r,az,el) pair from a user click on a snapshot (per camera)."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            data = request.get_json(force=True) or {}
            snap = data["snapshot_path"]
            u = float(data["u"]); v = float(data["v"])
            W = int(data.get("width", 640)); H = int(data.get("height", 480))
            FOVH = float(data.get("fov_h_deg", 90.0))
            sel_idx = int(data.get("selected_index", 0))

            # Prefer the per-snapshot sidecar (contains candidates + camera_id)
            side = os.path.join(SNAPSHOT_FOLDER, os.path.basename(snap)) + ".live.json"
            cam_id = None
            try:
                with open(side, "r") as f:
                    _sj = json.load(f) or {}
                cands = _sj.get("candidates", [])
                cam_id = _sj.get("camera_id")
            except Exception:
                cands = []
            if not cands:
                return jsonify({"ok": False, "error": "no radar candidates"}), 400

            if sel_idx < 0 or sel_idx >= len(cands):
                fx = (W/2.0)/math.tan(math.radians(FOVH/2.0)); cx=W/2.0
                az_est = math.degrees(math.atan2((u - cx), fx))
                sel_idx = min(range(len(cands)), key=lambda i: abs(float(cands[i].get("azimuth_deg", 0.0)) - az_est))

            c = cands[sel_idx]
            supports_cam = hasattr(add_pair, "__code__") and ("camera_id" in add_pair.__code__.co_varnames)
            kwargs = dict(
                u=u, v=v,
                r_m=float(c.get("distance_m", 0.0)),
                az_deg=float(c.get("azimuth_deg", 0.0)),
                el_deg=float(c.get("elevation_deg", 0.0)),
                meta={"source":"manual","snapshot_path":snap,"candidate_index":sel_idx,"camera_id": cam_id}
            )
            if supports_cam:
                kwargs["camera_id"] = cam_id
            add_pair(**kwargs)
            pairs = load_pairs(camera_id=cam_id) if (hasattr(load_pairs, "__code__")
                                                    and ("camera_id" in load_pairs.__code__.co_varnames)
                                                    and cam_id is not None) else load_pairs()
            return jsonify({"ok": True, "pairs": len(pairs)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/reload_candidates")
    @login_required
    def api_calibration_candidates():
        """Fetch latest radar candidates from the main service (no snapshot)."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            cands = _load_latest_candidates()
            return jsonify({"ok": True, "candidates": cands})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/pairs", methods=["GET"])
    @login_required
    def api_calibration_pairs():
        """List current (u,v)↔(r,az,el) pairs for the UI (per camera)."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            cam_id = request.args.get("camera_id")
            if hasattr(load_pairs, "__code__") and ("camera_id" in load_pairs.__code__.co_varnames) and cam_id is not None:
                pairs = load_pairs(camera_id=int(cam_id)) or []
            else:
                pairs = load_pairs() or []
            return jsonify({"ok": True, "count": len(pairs), "pairs": pairs})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/undo_last", methods=["POST"])
    @login_required
    def api_calibration_undo_last():
        """Remove the most recent pair for the selected camera."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            cam_id = (request.get_json(silent=True) or {}).get("camera_id") or request.args.get("camera_id")
            if hasattr(load_pairs, "__code__") and ("camera_id" in load_pairs.__code__.co_varnames) and cam_id is not None:
                pairs = load_pairs(camera_id=int(cam_id)) or []
            else:
                pairs = load_pairs() or []
            if not pairs:
                return jsonify({"ok": False, "error": "no pairs to remove"}), 409
            pairs.pop()
            if hasattr(save_pairs, "__code__") and ("camera_id" in save_pairs.__code__.co_varnames) and cam_id is not None:
                save_pairs(pairs, camera_id=int(cam_id))
            else:
                save_pairs(pairs)
            return jsonify({"ok": True, "count": len(pairs)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/fit", methods=["POST"])
    @login_required
    def api_calibration_fit():
        """
        Fit extrinsics from current pairs and stage the model; do not publish.
        UI can call this repeatedly until the reprojection error is acceptable.
        """
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            cfg = load_config()
            intr = estimate_intrinsics(
                int(cfg.get("camera_width_px", 640)),
                int(cfg.get("camera_height_px", 480)),
                float(cfg.get("camera_fov_h_deg", 90.0)),
                tuple(cfg.get("distortion", [])) if "distortion" in cfg else None
            )
            body = (request.get_json(silent=True) or {})
            cam_id = body.get("camera_id") or request.args.get("camera_id")
            kwargs = {"min_points": int(cfg.get("calibration_min_points", 12))}
            if hasattr(fit_extrinsics, "__code__") and ("camera_id" in fit_extrinsics.__code__.co_varnames) and cam_id is not None:
                kwargs["camera_id"] = int(cam_id)
            try:
                _, med_err, n_pairs = fit_extrinsics(intr, **kwargs)
            except RuntimeError as re:  # e.g., not enough pairs
                return jsonify({"ok": False, "error": str(re)}), 409
            return jsonify({"ok": True, "pairs": n_pairs, "median_px": round(float(med_err), 2)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/publish", methods=["POST"])
    @login_required
    def api_calibration_publish():
        """Publish the staged model as live (separate from fit)."""
        if not is_admin():
            return jsonify({"ok": False, "error": "Unauthorized"}), 403
        try:
            cam_id = (request.get_json(silent=True) or {}).get("camera_id") or request.args.get("camera_id")
            try:
                if hasattr(publish_model, "__code__") and ("camera_id" in publish_model.__code__.co_varnames) and cam_id is not None:
                    path = publish_model(camera_id=int(cam_id))
                else:
                    path = publish_model()
            except RuntimeError as re:  # e.g., "Nothing staged"
                return jsonify({"ok": False, "error": str(re)}), 409
            return jsonify({"ok": True, "path": path})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/calibration/status")
    @login_required
    def api_calibration_status():
        """Return per-camera calibration status (mode, pair count, staged/live presence, median px)."""
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            cfg = load_config()
            ok_px = float(cfg.get("calibration_reproj_ok_px", 6.0))
            mode = bool(cfg.get("calibration_mode", False))

            cam_id = request.args.get("camera_id")
            # Prefer per-camera paths if calibration._paths_for_camera exists
            try:
                from calibration import _paths_for_camera as _paths_for_camera_fn
            except Exception:
                _paths_for_camera_fn = None

            if _paths_for_camera_fn and cam_id is not None:
                # returns: (base_dir, pairs_path, live_path, staged_path)
                _, _pairs_path, live_path, staged_path = _paths_for_camera_fn(int(cam_id))
                pairs = (load_pairs(camera_id=int(cam_id))
                        if hasattr(load_pairs, "__code__") and "camera_id" in load_pairs.__code__.co_varnames
                        else load_pairs())
            else:
                # Fallback to legacy global paths
                _, live_path, staged_path, _ = _calib_paths()
                pairs = load_pairs()

            live_present = os.path.exists(live_path)
            staged_present = os.path.exists(staged_path)

            # Prefer median from LIVE model; fall back to staged
            median_px = None
            src_path = live_path if live_present else staged_path
            if src_path and os.path.exists(src_path):
                try:
                    with open(src_path, "r") as f:
                        sj = json.load(f)
                    meta = (sj.get("meta") or {})
                    median_px = (
                        meta.get("median_reproj_error_px")
                        or meta.get("median_reprojection_error_px")
                        or meta.get("median_px")
                    )
                    if isinstance(median_px, (int, float)):
                        median_px = round(float(median_px), 2)
                except Exception:
                    median_px = None

            staged_ok = (median_px is not None and median_px <= ok_px)
            return jsonify({
                "status": "ok",
                "calibration_mode": mode,
                "pairs": len(pairs),
                "staged": staged_present,
                "live": live_present,
                "model_present": live_present,
                "median_px": median_px,
                "staged_median_px": median_px,
                "ok_px": ok_px,
                "staged_ok": staged_ok
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    @app.route("/api/model_info")
    @login_required
    def api_model_info():
        # Always return normalized (0–100) so the tile can render immediately.
        meta = get_model_metadata() or {}
        acc = meta.get("accuracy")
        try:
            if acc is not None:
                a = float(acc)
                acc = (a * 100.0) if a <= 1.001 else a
        except Exception:
            acc = None
        out = {
            "accuracy": acc,
            "updated_at": meta.get("updated_at"),
            "method": meta.get("method"),
            "change": meta.get("change"),
        }
        return jsonify(out)
    
    # --- Model I/O helpers --------------------------------------------------------
    def _joblib_load_retry(path: str, attempts: int = 6, base_delay: float = 0.25):
        """
        Windows-safe loader: retry a few times in case AV/indexer briefly locks
        a file that was just swapped into place.
        """
        last = None
        for i in range(max(1, attempts)):
            try:
                return joblib.load(path)
            except Exception as e:
                last = e
                time.sleep(base_delay * (1.6 ** i))
        raise last

    def _write_json_atomic(path: str, obj: dict):
        """Write JSON sidecar atomically (prevents partial reads)."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    @app.route("/retrain_model", methods=["POST", "OPTIONS"])
    @app.route("/api/model/retrain", methods=["POST", "OPTIONS"])
    @login_required
    def api_retrain_model():
        """
        Retrain LightGBM model and publish fresh metadata.
        (Fixes: allowed methods, accuracy parsing, artifact shape handling.)
        """
        if request.method == "OPTIONS":
            return ("", 204)
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        try:
            # Run trainer with unbuffered stdout so the UI can read logs as they happen.
            trainer = os.path.join(BASE_DIR, "train_lightbgm.py")
            cmd = [sys.executable, "-u", trainer]
            res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
            if res.returncode != 0:
                out = (res.stdout or "").strip()
                err = (res.stderr or "").strip()
                # Try to surface a friendly abort reason
                reason = None
                for line in out.splitlines():
                    ls = line.strip()
                    if ls.startswith("[ABORT]") or ls.startswith("ABORT_REASON:"):
                        reason = ls; break
                    if "No training rows" in ls or "All labels are UNKNOWN" in ls:
                        reason = ls; break
                return jsonify({
                    "error": "train_failed",
                    "reason": reason,
                    "stdout": out[-4000:],   # cap response size
                    "stderr": err[-4000:]
                }), 500

            # Freshly written model
            model_path = os.path.join(BASE_DIR, "radar_lightgbm_model.pkl")
            payload = _joblib_load_retry(model_path)

            # Accept both (model, scaler) and dict artifacts
            model = scaler = None
            features = []
            labels = []
            if isinstance(payload, tuple) and len(payload) >= 1:
                model = payload[0]
                scaler = payload[1] if len(payload) > 1 else None
                try:
                    features = list(getattr(model, "feature_name_", []) or [])
                    labels = list(getattr(model, "classes_", []) or [])
                except Exception:
                    pass
            elif isinstance(payload, dict):
                model    = payload.get("model")
                scaler   = payload.get("scaler")
                features = list(payload.get("features") or [])
                labels   = list(payload.get("labels") or [])

            # Parse accuracy from stdout; accept normalized or % formats
            acc = None
            try:
                m = (re.search(r"accuracy\\s*[:=]\\s*([0-9.]+)\\s*%", res.stdout or "", re.I)
                     or re.search(r"\\baccuracy\\s*[:=]\\s*([0-9.]+)", res.stdout or "", re.I))
                if m:
                    v = float(m.group(1))
                    # If it's clearly a percentage (e.g., 87.12), normalize to 0.8712; else keep 0-1
                    acc = (v/100.0) if v > 1.001 else v
            except Exception:
                acc = None

            # Version & DB metadata
            try:
                version = _next_model_version()
                save_model_metadata_db(version, features, acc, labels)
            except Exception as e:
                logger.warning(f"[MODEL META] save skipped: {e}")

            # Sidecar for quick reads
            try:
                _write_json_atomic(os.path.join(BASE_DIR, "model_metadata.json"), {
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "accuracy": round(acc, 4) if acc is not None else None,
                    "source": "retrain"
                })
            except Exception as e:
                logger.debug(f"[MODEL SIDECAR] write skipped: {e}")

            # UI change table
            try:
                save_model_info(acc, "retrain")
            except Exception:
                pass

            return jsonify({
                "ok": True,
                "model_path": model_path,
                "accuracy": acc,
                "features": features,
                "labels": labels
            })
        except Exception as e:
            logger.exception("[RETRAIN] failed")
            return jsonify({"error": f"{type(e).__name__}", "message": str(e)}), 500

    @app.route("/upload_model", methods=["POST"])
    @login_required
    def upload_model():
        """
        Validate and install an uploaded model file.
        (Fixes: accept both dict artifact and (model, scaler) tuple; persist metadata.)
        """
        if not is_admin():
            return jsonify({"error": "Unauthorized"}), 403
        if "model_file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["model_file"]
        if not file.filename.endswith(".pkl"):
            return jsonify({"error": "File must be a .pkl"}), 400
        try:
            with NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)
            payload = joblib.load(tmp_path)
            os.unlink(tmp_path)
            # Normalize to dict artifact
            if isinstance(payload, tuple):
                model, scaler = payload[0], (payload[1] if len(payload) > 1 else None)
                artifact = {"model": model, "scaler": scaler}
            elif isinstance(payload, dict):
                artifact = payload
                model = artifact.get("model"); scaler = artifact.get("scaler")
            else:
                return jsonify({"error": "Unsupported artifact format"}), 400

            # Light type checks (avoid hard dependency on class names)
            if model is None:
                return jsonify({"error": "Missing model"}), 400

            # Persist canonical filename
            out_path = os.path.join(BASE_DIR, "radar_lightgbm_model.pkl")
            joblib.dump(artifact, out_path)
            logger.info("[MODEL] Uploaded model saved to radar_lightgbm_model.pkl")

            # Derive labels/features if present
            labels = list(artifact.get("labels") or []) or list(getattr(model, "classes_", []) or [])
            features = list(artifact.get("features") or []) or list(getattr(model, "feature_name_", []) or [])

            version = _next_model_version()
            logger.info(f"[MODEL] Saving metadata: version={version}, source=upload, labels={len(labels)}, features={len(features)}")
            save_model_metadata_db(version, features, None, labels)  # accuracy unknown for uploads
            save_model_info(None, "upload")
            _write_json_atomic(os.path.join(BASE_DIR, "model_metadata.json"), {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": None,
                "source": "upload"
            })
            return jsonify({
                "status": "ok",
                "message": "Model uploaded and validated successfully.",
                "version": version,
                "labels": labels
            })
        except Exception as e:
            logger.exception("[MODEL UPLOAD ERROR]")
            return jsonify({"error": "Upload failed", "details": str(e)}), 500
        
    def save_model_sidecar(accuracy, source):
        """Sidecar cache for quick reads and debugging; not authoritative."""
        metadata = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "source": source
        }
        try:
            with open("model_metadata.json", "w") as f:
                json.dump(metadata, f)
            logger.info(f"[MODEL SIDEcar] Saved: {metadata}")
        except Exception as e:
            logger.error(f"[MODEL SIDEcar] Failed to save metadata: {e}")

    def _load_model_and_feature_importance():
        """
        Returns (feature_names:list[str], importances:list[float]) from the saved model.
        Falls back gracefully if anything is missing.
        """
        names, imps = [], []
        try:
            model_path = os.path.join(os.path.dirname(__file__), "radar_lightgbm_model.pkl")
            payload = joblib.load(model_path)
            # Accept both dict artifact and tuple
            if isinstance(payload, dict):
                model = payload.get("model")
            else:
                model = payload[0]
            # Feature names
            names = list(getattr(model, "feature_name_", []) or [])
            if not names and hasattr(model, "booster_"):
                try:
                    names = list(model.booster_.feature_name())
                except Exception:
                    names = []
            # Importances
            imps = getattr(model, "feature_importances_", None)
            if imps is None and hasattr(model, "booster_"):
                try:
                    imps = model.booster_.feature_importance(importance_type="gain")
                except Exception:
                    imps = model.booster_.feature_importance()
            imps = [] if imps is None else [float(v) for v in imps]

            # Align lengths conservatively
            if names and len(names) != len(imps):
                names = [str(n) for n in names[:len(imps)]]
            if not names and imps:
                names = [f"f{i}" for i in range(len(imps))]
        except Exception as e:
            logger.warning(f"[FI] Load model/feature importance failed: {e}")
        return names, imps

    def get_model_sidecar():
        path = "radar_lightgbm_model.pkl"
        meta_path = "model_metadata.json"

        if not os.path.exists(path):
            return {"status": "Model not found"}

        try:
            payload = joblib.load(path)
            if isinstance(payload, dict):
                model = payload.get("model")
            else:
                model = payload[0]
            classes = list(model.classes_) if hasattr(model, "classes_") else []

            metadata = {
                "status": "OK",
                "trained_classes": classes,
                "accuracy": None,
                "updated_at": None,
                "method": None,
                "change": None
            }

            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    try:
                        meta_data_file = json.load(f)
                        metadata.update(meta_data_file)
                    except Exception as e:
                        logger.warning(f"Corrupt model metadata file: {e}")
            return metadata
        except Exception as e:
            logger.error(f"[MODEL SIDEcar] Load error: {e}")
            return {"status": "Error", "error": str(e)}

    @app.route("/api/model_sidecar", methods=["GET"])
    @login_required
    def api_model_sidecar():
        try:
            try:
                side = get_model_sidecar() if 'get_model_sidecar' in globals() else {}
            except Exception:
                side = {}

            meta = None
            try:
                meta = get_latest_model_metadata_db()
            except Exception:
                meta = None

            def _mget(m, key, idx):
                if m is None: return None
                # DictRow behaves like mapping; fallback to tuple idx
                try:
                    if hasattr(m, "keys"):
                        return m.get(key) if hasattr(m, "get") else m[key]
                except Exception:
                    pass
                try:
                    return m[idx]
                except Exception:
                    return None

            out = {
                "status": "ok",
                "sidecar": side or {},
                "version": _mget(meta, "version", 1),
                "features": _mget(meta, "features", 2) or [],
                "labels": _mget(meta, "labels", 4) or [],
            }
            out["n_features"] = len(out["features"])
            out["n_labels"] = len(out["labels"])
            return jsonify(out)
        except Exception as e:
            logger.exception("[MODEL SIDECAR] error")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/model_feature_importance", methods=["GET"])
    @login_required
    def api_model_feature_importance():
        try:
            path = "radar_lightgbm_model.pkl"
            if not os.path.exists(path):
                return jsonify({"error": "Model file not found"}), 404
            payload = joblib.load(path)
            if isinstance(payload, dict):
                model = payload.get("model")
            else:
                model = payload[0]
            if model is None:
                return jsonify({"error": "Model missing"}), 400
            # importances
            imp = getattr(model, "feature_importances_", None)
            if imp is None and hasattr(model, "booster_"):
                try:
                    imp = model.booster_.feature_importance(importance_type="gain")
                except Exception:
                    imp = model.booster_.feature_importance()
            if imp is None:
                return jsonify({"error": "Model has no importances"}), 400
            imp = [float(x) for x in list(imp)]

            def _maybe_names_from_db(expected_len: int):
                try:
                    row = get_latest_model_metadata_db()
                    arr = (row.get("features") if hasattr(row, "get") else row[2]) or []
                    return list(arr) if len(arr) == expected_len else None
                except Exception:
                    return None

            names = list(getattr(model, "feature_name_", []) or [])
            generic = (not names) or all(str(n).lower().startswith(("feature", "column_", "f")) for n in names)
            if generic or len(names) != len(imp):
                dbn = _maybe_names_from_db(len(imp))
                if dbn: names = dbn
            if not names:
                names = [f"f{i}" for i in range(len(imp))]
            pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:20]
            return jsonify({"status":"ok", "features":[n for n,_ in pairs], "importance":[v for _,v in pairs]})
        except Exception as e:
            logger.exception("[MODEL] feature importance error")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/model_feature_importance_image")
    @login_required
    def api_model_feature_importance_image():
        """
        Lightweight PNG (top-N) so the UI can show FI without hanging.
        Example: /api/model_feature_importance_image?top=20
        """
        try:
            top = int(request.args.get("top", "15"))
            top = max(1, min(top, 30))
        except Exception:
            top = 15

        names, imps = _load_model_and_feature_importance()

        # Render a tiny, headless PNG either with the data or a friendly message.
        fig_h = max(1.6, 0.35 * min(top, len(names) or 1))
        fig, ax = plt.subplots(figsize=(7, fig_h), dpi=120)

        if names and imps:
            pairs = sorted(zip(names, imps), key=lambda x: x[1], reverse=True)[:top]
            labels = [p[0] for p in pairs]
            vals = [float(p[1]) for p in pairs]
            y = list(range(len(labels)))
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Importance (gain)")
            ax.set_title("LightGBM Feature Importance", fontsize=11, pad=6)
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No feature importance available",
                    ha="center", va="center", fontsize=11)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png",
                        as_attachment=False,
                        download_name="feature_importance.png")


    @app.route("/download/feature_importance.csv")
    @login_required
    def download_feature_importance_csv():
        """
        CSV export using *feature names* instead of column numbers.
        Columns: feature,importance (sorted desc).
        """
        names, imps = _load_model_and_feature_importance()
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["feature", "importance"])
        if names and imps and len(names) == len(imps):
            for n, v in sorted(zip(names, imps), key=lambda x: x[1], reverse=True):
                w.writerow([n, f"{float(v):.6f}"])
        data = out.getvalue().encode("utf-8")
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=feature_importance.csv"}
        )

    @app.route("/api/status")
    @login_required
    def api_status():
        try:
            cfg = load_config()
            ports = (cfg.get("iwr_ports") or {})
            cli = ports.get("cli")
            data = ports.get("data")

            cli_ok = bool(check_radar_connection(cli, 115200, timeout=0.6)) if cli else False
            data_ok = bool(check_radar_connection(data, 921600, timeout=0.6)) if data else False

            # close the handles returned by check_radar_connection
            if isinstance(cli_ok, object) and hasattr(cli_ok, "close"):
                try: cli_ok.close()
                except: pass
                cli_ok = True
            if isinstance(data_ok, object) and hasattr(data_ok, "close"):
                try: data_ok.close()
                except: pass
                data_ok = True

            radar_status = "connected" if (cli_ok and data_ok) else ("partial" if (cli_ok or data_ok) else "disconnected")

            # Camera considered “connected” if a camera is enabled in config/DB
            cameras, selected = load_cameras_from_db()
            cam = cameras[selected] if cameras else {}
            camera_connected = bool(cam.get("enabled", True) and (cam.get("url") or cam.get("stream_type")))

            # Storage
            disk = psutil.disk_usage("/")
            storage_free_gb = round(disk.free / (1024**3), 2)

            h = _fetch_pi_health()
            if h.get("ok") and h.get("radar_connected") is not None:
                radar_flag = "connected" if h.get("radar_connected") else "disconnected"
            else:
                radar_flag = radar_status  # existing logic

            return jsonify({
                "status": "ok",
                "radar": radar_flag,
                "camera": "connected" if camera_connected else "disconnected",
                "storage_gb_free": storage_free_gb,
                "timestamp": datetime.now(IST).isoformat(),
                "pi_health": h
            })
        except Exception as e:
            logger.error(f"[STATUS] {e}")
            return jsonify({"status": "error", "error": str(e)}), 500
    
    @app.route("/control", methods=["GET", "POST"])
    @login_required
    def control():
        if not is_admin():
            flash("Admin access required", "error")
            return redirect(url_for("index"))
        
        message = None
        config = load_config()
        try:
            cameras, selected = load_cameras_from_db()
            for cam in cameras:
                if "stream_type" not in cam:
                    cam["stream_type"] = "mjpeg"  # default fallback
            config["cameras"] = cameras
            config["selected_camera"] = selected
        except Exception as e:
            logger.warning(f"Could not load cameras from DB: {e}")
            config["cameras"] = []
            config["selected_camera"] = 0
        snapshot = None
        
        if request.method == "POST":
            action = request.form.get("action")
            
            try:
                if action == "clear_db":
                    with get_db_connection() as conn, conn.cursor() as cur:
                        cur.execute("TRUNCATE radar_data RESTART IDENTITY")
                        conn.commit()
                    message = "All radar data cleared successfully."

                elif action == "run_calibration":
                    try:
                        cfg = load_config()
                        # toggle on/off
                        if not cfg.get("calibration_mode", False):
                            cfg["calibration_mode"] = True
                            save_config(cfg)
                            open("reload_flag.txt", "w").close()
                            reset_pairs()
                            message = "Calibration mode ENABLED. The system will start collecting pairs."
                        else:
                            cfg["calibration_mode"] = False
                            save_config(cfg)
                            open("reload_flag.txt", "w").close()
                            message = "Calibration mode DISABLED."
                        flash(message, "success")
                    except Exception as e:
                        logger.error(f"[CALIB] toggle failed: {e}")
                        flash(f"Calibration toggle failed: {e}", "error")
                        message = f"Calibration toggle failed: {e}"

                elif action == "fit_projection":
                    try:
                        cfg = load_config()
                        intr = estimate_intrinsics(
                            int(cfg.get("camera_width_px", 640)),
                            int(cfg.get("camera_height_px", 480)),
                            float(cfg.get("camera_fov_h_deg", 90.0)),
                            tuple(cfg.get("distortion", [])) if "distortion" in cfg else None
                        )
                        model, med_err, n = fit_extrinsics(intr, min_points=int(cfg.get("calibration_min_points", 12)))
                        ok_px = float(cfg.get("calibration_reproj_ok_px", 6.0))
                        message = f"Staged model from {n} pairs, median error = {med_err:.2f}px"
                        if med_err <= ok_px:
                            path = publish_model()
                            message += f" — PUBLISHED ({path})."
                        flash(message, "success")
                    except Exception as e:
                        logger.error(f"[CALIB FIT] {e}")
                        flash(f"Calibration fit failed: {e}", "error")
                        message = f"Calibration fit failed: {e}"
                    
                elif action == "backup_db":
                    try:
                        backup_name = f"radar_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dump"
                        backup_path = os.path.join(BACKUP_FOLDER, backup_name)
                        os.makedirs(BACKUP_FOLDER, exist_ok=True)

                        pg_dump = (os.getenv("PG_DUMP")
                                   or shutil.which("pg_dump")
                                   or shutil.which("pg_dump.exe"))
                        if not pg_dump:
                            raise FileNotFoundError("pg_dump not found. Install PostgreSQL client tools or set PG_DUMP.")

                        uri = _dsn_to_uri(DB_DSN)
                        if not uri:
                            raise RuntimeError("DB_DSN not set")
                        result = subprocess.run(
                            [pg_dump, "-d", uri, "-F", "c", "-f", backup_path],
                            capture_output=True, text=True, check=True
                        )

                        return send_file(backup_path, as_attachment=True, download_name=backup_name)

                    except subprocess.CalledProcessError as e:
                        logger.error(f"[BACKUP ERROR] pg_dump failed\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
                        message = f"Database backup failed. Reason: {e.stderr.strip()}"
                    except Exception as e:
                        logger.error(f"[BACKUP ERROR] Unexpected: {e}")
                        message = f"Unexpected error during backup: {e}"
                    
                elif action == "restore_db":
                    if 'backup_file' in request.files:
                        file = request.files['backup_file']
                        if file:
                            filename = secure_filename(file.filename)
                            temp_path = os.path.join(BACKUP_FOLDER, f"temp_{filename}")
                            file.save(temp_path)

                            try:
                                uri = _dsn_to_uri(DB_DSN)
                                if not uri:
                                    raise RuntimeError("DB_DSN not set")
                                is_custom = filename.lower().endswith((".dump", ".backup"))
                                if is_custom:
                                    pg_restore = (os.getenv("PG_RESTORE")
                                                  or shutil.which("pg_restore")
                                                  or shutil.which("pg_restore.exe"))
                                    if not pg_restore:
                                        raise FileNotFoundError("pg_restore not found. Install PostgreSQL client tools or set PG_RESTORE.")
                                    # Cleanly drop & recreate objects if they exist (handles enum types safely)
                                    cmd = [
                                        pg_restore, "-d", uri,
                                        "--clean", "--if-exists",
                                        "--no-owner", "--no-privileges",
                                        "--single-transaction", "-v",
                                        temp_path
                                    ]
                                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                                else:
                                    # Legacy .sql: best-effort handling; don't abort on harmless "already exists"
                                    psql = (os.getenv("PSQL_BIN")
                                            or shutil.which("psql")
                                            or shutil.which("psql.exe"))
                                    if not psql:
                                        raise FileNotFoundError("psql not found. Install PostgreSQL client tools or set PSQL_BIN.")
                                    # Try a tolerant first pass to precreate missing types if needed
                                    preface = """
                                        DO $$
                                        BEGIN
                                          IF EXISTS (SELECT 1 FROM pg_type WHERE typname='camera_stream_type') THEN
                                            DROP TYPE camera_stream_type;
                                          END IF;
                                        EXCEPTION WHEN others THEN
                                         -- ignore
                                        END$$;
                                    """
                                    subprocess.run([psql, uri, "-v", "ON_ERROR_STOP=0", "-c", preface],
                                                   capture_output=True, text=True, check=False)
                                    # Now run the script with ON_ERROR_STOP=1
                                    result = subprocess.run(
                                        [psql, uri, "-v", "ON_ERROR_STOP=1", "-f", temp_path],
                                        capture_output=True, text=True, check=True
                                    )
                                os.remove(temp_path)

                                # Post-restore validation using the app pool
                                try:
                                    with get_db_connection() as conn, conn.cursor() as cur:
                                        cur.execute("SELECT COUNT(*) FROM radar_data")
                                        row_count = cur.fetchone()[0]
                                    if row_count == 0:
                                        message = "Restore completed, but no data was found. Check if your backup includes INSERTs or use a .dump made with pg_dump -Fc."
                                    else:
                                        message = f"Database restored successfully with {row_count} rows."
                                except Exception as e:
                                    logger.warning(f"[POST-RESTORE VALIDATION ERROR] {e}")
                                    message = "Restore finished, but failed to verify data count."

                            except subprocess.CalledProcessError as e:
                                logger.error(f"[RESTORE ERROR] Failed\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
                                message = f"Restore failed: {e.stderr.strip()}"
                            except Exception as e:
                                logger.error(f"[RESTORE ERROR] Unexpected: {e}")
                                message = f"Unexpected error during restore: {e}"
                        else:
                            message = "Please upload a .dump (preferred) or .sql backup."
                            
                elif action == "cleanup_snapshots":
                    # Robust retention cleanup: DB timestamps OR file age OR orphans.
                    retention_days = int(request.form.get("retention_days", config.get("retention_days", 30)))
                    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)

                    deleted_files = 0
                    deleted_records = 0
                    removed_bundles = 0

                    def _is_older_than_cutoff_path(p: str) -> bool:
                        try:
                            rp = os.path.realpath(os.path.normpath(p))
                            if os.path.exists(rp):
                                mt = datetime.fromtimestamp(os.path.getmtime(rp), tz=timezone.utc)
                                return mt < cutoff
                        except Exception:
                            pass
                        return False

                    with get_db_connection() as conn:
                        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                        # 1) Primary selection by DB age (as before)
                        cur.execute("""
                            SELECT id, snapshot_path
                              FROM radar_data
                             WHERE (
                                     measured_at IS NOT NULL AND measured_at < %s
                                   )
                                OR (
                                     measured_at IS NULL
                                 AND COALESCE(NULLIF(TRIM(COALESCE(datetime::text, '')), ''), '') <> ''
                                 AND COALESCE(
                                         to_timestamp(NULLIF(TRIM(COALESCE(datetime::text,'')),'')::double precision),
                                         (NULLIF(TRIM(COALESCE(datetime::text,'')),'')::timestamptz)
                                     ) < %s
                                   )
                        """, (cutoff, cutoff))
                        rows = list(cur.fetchall())

                        # 2) Filesystem-age fallback: add rows whose files/bundles are older than cutoff
                        cur.execute("""SELECT id, snapshot_path FROM radar_data WHERE snapshot_path IS NOT NULL""")
                        all_with_paths = list(cur.fetchall())
                        id_already = {r["id"] for r in rows}
                        for r in all_with_paths:
                            if r["id"] in id_already:
                                continue
                            p = (r.get("snapshot_path") or "").strip()
                            if not p:
                                continue
                            # consider both snapshot file mtime and bundle dir (if resolvable)
                            older = _is_older_than_cutoff_path(p)
                            if not older:
                                bdir = _guess_bundle_from_snapshot_path(p)
                                if bdir:
                                    older = _is_older_than_cutoff_path(bdir)
                            if older:
                                rows.append(r)

                        # 3) Delete files/bundles for selected rows
                        for r in rows:
                            p = (r.get("snapshot_path") or "").strip()
                            if not p:
                                continue
                            try:
                                rp = os.path.realpath(os.path.normpath(p))
                            except Exception:
                                rp = p
                            if "violations" in (rp or ""):
                                bdir = os.path.dirname(rp) if os.path.basename(rp) != "image.jpg" else os.path.dirname(rp)
                                # Ensure we point to the actual bundle dir
                                if os.path.basename(rp) == "image.jpg":
                                    bdir = os.path.dirname(rp)
                                else:
                                    guess = _guess_bundle_from_snapshot_path(rp)
                                    bdir = guess or os.path.dirname(rp)
                                if os.path.isdir(bdir):
                                    try:
                                        shutil.rmtree(bdir)
                                        removed_bundles += 1
                                    except Exception as e:
                                        logger.warning(f"[CLEANUP] Failed to remove bundle {bdir}: {e}")
                                else:
                                    if os.path.isfile(rp):
                                        try:
                                            os.remove(rp); deleted_files += 1
                                        except Exception as e:
                                            logger.warning(f"[CLEANUP] Failed to delete {rp}: {e}")
                            else:
                                if rp and os.path.isfile(rp):
                                    try:
                                        os.remove(rp); deleted_files += 1
                                    except Exception as e:
                                        logger.warning(f"[CLEANUP] Failed to delete {rp}: {e}")

                        # 4) Delete DB rows for selected ids
                        if rows:
                            ids = [int(r["id"]) for r in rows]
                            cur.execute("DELETE FROM radar_data WHERE id = ANY(%s)", (ids,))
                            deleted_records = cur.rowcount
                            conn.commit()

                    # 5) Orphan sweep: leftover old files with no DB row
                    try:
                        # snapshots/
                        for name in os.listdir(SNAPSHOT_FOLDER):
                            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                                continue
                            fp = os.path.join(SNAPSHOT_FOLDER, name)
                            if not _is_older_than_cutoff_path(fp):
                                continue
                            # if no DB row references this basename, delete file
                            with get_db_connection() as conn:
                                row = _row_by_snapshot_basename(conn, name)
                            if not row:
                                if _safe_unlink(fp):
                                    deleted_files += 1
                        # violations/ dated subfolders
                        if os.path.isdir(BUNDLES_DIR):
                            for day in os.listdir(BUNDLES_DIR):
                                ddir = os.path.join(BUNDLES_DIR, day)
                                if not os.path.isdir(ddir):
                                    continue
                                for b in os.listdir(ddir):
                                    bdir = os.path.join(ddir, b)
                                    try:
                                        mt = datetime.fromtimestamp(os.path.getmtime(bdir), tz=timezone.utc)
                                    except Exception:
                                        mt = None
                                    if mt and mt < cutoff:
                                        try:
                                            shutil.rmtree(bdir)
                                            removed_bundles += 1
                                        except Exception:
                                            pass
                    except Exception as e:
                        logger.debug(f"[CLEANUP ORPHANS] {e}")

                    message = f"Cleaned up {deleted_files} snapshots and {deleted_records} records; removed {removed_bundles} bundles."
                                            
                elif action == "test_radar":
                    try:
                        # 1) Prefer Pi health HTTP if configured
                        tried_http = False
                        http_ok = False
                        if PI_HEALTH_URL:
                            tried_http = True
                            health_url = PI_HEALTH_URL.strip()
                            if not health_url.endswith("/health"):
                                health_url = health_url.rstrip("/") + "/health"
                            try:
                                r = requests.get(health_url, timeout=2.5)
                                if r.status_code == 200:
                                    j = r.json()
                                    if bool(j.get("radar_connected", False)):
                                        http_ok = True
                                        message = "Radar test successful (Pi health reports connected)."
                                        flash(message, "success")
                            except Exception as he:
                                logger.warning(f"[RADAR TEST] Pi health probe failed: {he}")

                        if not http_ok:
                            # 2) Fallback to local UART probe (works when running on the Pi)
                            cfg = load_config()
                            ports = (cfg.get("iwr_ports") or {})
                            cli = ports.get("cli")
                            data = ports.get("data")

                            cli_ser  = check_radar_connection(cli, 115200, timeout=0.6) if cli else None
                            data_ser = check_radar_connection(data, 921600, timeout=0.6) if data else None

                            # tidy up
                            if cli_ser:
                                try: cli_ser.close()
                                except: pass
                            if data_ser:
                                try: data_ser.close()
                                except: pass

                            if cli_ser and data_ser:
                                message = "Radar test successful. Both UART ports opened."
                                flash(message, "success")
                            elif cli_ser or data_ser:
                                which = "CLI" if cli_ser else "DATA"
                                message = f"Partial radar test passed. {which} port opened; check cable/permissions for the other."
                                flash(message, "warning")
                            else:
                                if tried_http:
                                    message = "Radar test failed. Pi health not connected and UART probe also failed."
                                else:
                                    message = "Radar test failed. Could not open either UART port."
                                flash(message, "error")
                    except Exception as e:
                        logger.error(f"[RADAR TEST] {e}")
                        message = f"Radar test error: {e}"
                        flash(message, "error")
                    
                elif action == "test_camera":
                    return redirect(url_for("cam_test"))
                    
                elif action == "update_config":
                    config["cooldown_seconds"] = float(request.form.get("cooldown_seconds", 0.5))
                    config["retention_days"] = int(request.form.get("retention_days", 30))
                    config["selected_camera"] = int(request.form.get("selected_camera", 0))
                    config["annotation_conf_threshold"] = float(request.form.get("annotation_conf_threshold", 0.5))
                    config["label_format"] = request.form.get("label_format", "{type} | {speed:.1f} km/h")

                    # ── Site & Location (map) ──
                    site = dict(config.get("site") or {})
                    site["site_id"] = (request.form.get("site_id") or site.get("site_id") or "").strip()
                    site["name"]    = (request.form.get("site_name") or site.get("name") or "").strip()
                    site["address"] = (request.form.get("site_address") or site.get("address") or "").strip()
                    tz_in = (request.form.get("site_timezone") or site.get("timezone") or "").strip()
                    if tz_in:
                        site["timezone"] = tz_in
                    # coordinates (optional casts keep previous on bad input)
                    lat_in = request.form.get("site_lat")
                    lon_in = request.form.get("site_lon")
                    try:
                        if lat_in not in (None, ""):
                            site["lat"] = float(lat_in)
                        if lon_in not in (None, ""):
                            site["lon"] = float(lon_in)
                    except ValueError:
                        pass
                    config["site"] = site

                    loc = dict(config.get("location") or {})
                    loc["enabled"]    = bool(request.form.get("location_enabled"))
                    loc["prefer_gps"] = bool(request.form.get("prefer_gps"))
                    gpsd_host_in = (request.form.get("gpsd_host") or loc.get("gpsd_host") or "").strip()
                    if gpsd_host_in:
                        loc["gpsd_host"] = gpsd_host_in
                    try:
                        gpsd_port_in = request.form.get("gpsd_port")
                        if gpsd_port_in not in (None, ""):
                            loc["gpsd_port"] = int(gpsd_port_in)
                        poll_in = request.form.get("poll_seconds")
                        if poll_in not in (None, ""):
                            loc["poll_seconds"] = max(1, int(poll_in))
                    except ValueError:
                        pass
                    config["location"] = loc

                # Parse all camera rows (multi-camera)
                cameras = []
                primary_idx = None
                # collect exactly the indices the form posted
                posted = []
                for v in request.form.getlist("camera_index"):
                    try:
                        posted.append(int(v))
                    except Exception:
                        continue
                for i in sorted(set(posted)):
                    # skip items explicitly marked for deletion
                    if str(request.form.get(f"camera_delete_{i}", "0")).lower() in ("1","true","on","yes"):
                        continue
                    cam_url      = (request.form.get(f"camera_url_{i}") or "").strip()
                    snap_url     = (request.form.get(f"camera_snapshot_url_{i}") or "").strip()
                    # ignore empty rows (neither RTSP/MJPEG nor snapshot given)
                    if not cam_url and not snap_url:
                        continue
                    raw_role     = (request.form.get(f"camera_role_{i}", "aux") or "aux").strip().lower()
                    if raw_role == "primary":
                        primary_idx = i
                    role_for_db  = raw_role if raw_role in ("primary","aux","ptz") else "aux"
                    cameras.append({
                        "id":           request.form.get(f"camera_id_{i}", f"cam{i}"),
                        "name":         request.form.get(f"camera_name_{i}", f"Camera {i+1}"),
                        "url":          cam_url,
                        "snapshot_url": snap_url,
                        "username":     (request.form.get(f"camera_username_{i}") or "").strip(),
                        "password":     (request.form.get(f"camera_password_{i}") or "").strip(),
                        "stream_type":  (request.form.get(f"camera_stream_type_{i}", "mjpeg") or "mjpeg").strip().lower(),
                        "role":         role_for_db,
                        "enabled":      (
                            True if request.form.get(f"camera_enabled_{i}") is None
                            else str(request.form.get(f"camera_enabled_{i}")).lower() in ("1","true","on","yes")
                        ),
                    })

                if cameras:
                    seen_primary = False
                    for idx, c in enumerate(cameras):
                        if c.get("role") == "primary":
                            if not seen_primary:
                                seen_primary = True
                            else:
                                c["role"] = "aux"
                    config["cameras"] = cameras
                    # one—and only one—Primary: use UI-chosen primary when present
                    if primary_idx is not None:
                        sel_idx = int(primary_idx)
                    else:
                        sel_idx = int(request.form.get("selected_camera_idx", config.get("selected_camera", 0)) or 0)
                    config["selected_camera"] = max(0, min(sel_idx, len(cameras)-1))
                    save_cameras_to_db(cameras, config.get("selected_camera", 0))
                    save_config(config)

                    # Dynamic speed limits
                    updated_limits = {}
                    for key in config.get("dynamic_speed_limits", {}).keys():
                        form_key = f"speed_limit_{key}"
                        val = request.form.get(form_key)
                        if val:
                            try:
                                updated_limits[key] = float(val)
                            except ValueError:
                                pass  # retain old

                    if updated_limits:
                        config["dynamic_speed_limits"] = updated_limits

                    # ── PTZ configuration (connection, mount, home, control, zoom, intrinsics) ──
                    try:
                        ptz = dict(config.get("ptz") or {})

                        def _maybe_set(dst: dict, key: str, form_key: str, caster=str):
                            v = request.form.get(form_key)
                            if v is None or (isinstance(v, str) and v.strip() == ""):
                                return
                            try:
                                dst[key] = caster(v)
                            except Exception:
                                # keep previous value on cast error
                                pass

                        # Connection / auth
                        _maybe_set(ptz, "host",          "ptz_host", str)
                        _maybe_set(ptz, "port",          "ptz_port", int)
                        _maybe_set(ptz, "scheme",        "ptz_scheme", str)          # "http" | "https"
                        _maybe_set(ptz, "auth_mode",     "ptz_auth_mode", str)       # "basic" | "digest" | "none"
                        _maybe_set(ptz, "username",      "ptz_username", str)
                        _maybe_set(ptz, "password",      "ptz_password", str)
                        _maybe_set(ptz, "profile_token", "ptz_profile_token", str)   # ONVIF profile name/token (optional)

                        # Mount (camera to radar transform / installation offsets)
                        mount = dict(ptz.get("mount") or {})
                        _maybe_set(mount, "dx_m",     "ptz_mount_dx_m", float)
                        _maybe_set(mount, "dy_m",     "ptz_mount_dy_m", float)
                        _maybe_set(mount, "dz_m",     "ptz_mount_dz_m", float)
                        _maybe_set(mount, "yaw_deg",  "ptz_mount_yaw_deg", float)
                        _maybe_set(mount, "pitch_deg","ptz_mount_pitch_deg", float)
                        _maybe_set(mount, "roll_deg", "ptz_mount_roll_deg", float)
                        ptz["mount"] = mount

                        # Home pose / preset
                        home = dict(ptz.get("home") or {})
                        _maybe_set(home, "pan_deg",   "ptz_home_pan_deg", float)
                        _maybe_set(home, "tilt_deg",  "ptz_home_tilt_deg", float)
                        _maybe_set(home, "zoom",      "ptz_home_zoom", float)
                        _maybe_set(home, "preset",    "ptz_home_preset", str)
                        home["wait_settle"] = bool(request.form.get("ptz_home_wait_settle")) \
                            if ("ptz_home_wait_settle" in request.form) else home.get("wait_settle", True)
                        _maybe_set(home, "settle_s",               "ptz_home_settle_s", float)
                        _maybe_set(home, "settle_samples",         "ptz_home_settle_samples", int)
                        _maybe_set(home, "settle_sample_sleep_s",  "ptz_home_settle_sample_sleep_s", float)
                        ptz["home"] = home

                        # Control gains / limits
                        ctrl = dict(ptz.get("control") or {})
                        _maybe_set(ctrl, "deadband_deg", "ptz_deadband_deg", float)
                        _maybe_set(ctrl, "k_pan",        "ptz_k_pan", float)
                        _maybe_set(ctrl, "k_tilt",       "ptz_k_tilt", float)
                        _maybe_set(ctrl, "max_vx",       "ptz_max_vx", float)
                        _maybe_set(ctrl, "max_vy",       "ptz_max_vy", float)
                        _maybe_set(ctrl, "max_step_s",   "ptz_max_step_s", float)
                        _maybe_set(ctrl, "stop_pulse_s", "ptz_stop_pulse_s", float)
                        ptz["control"] = ctrl

                        # Zoom constraints
                        zoom = dict(ptz.get("zoom") or {})
                        _maybe_set(zoom, "min", "ptz_zoom_min", float)
                        _maybe_set(zoom, "max", "ptz_zoom_max", float)
                        ptz["zoom"] = zoom

                        # Intrinsics (used for aiming/fov math, if applicable)
                        intr = dict(ptz.get("intrinsics") or {})
                        _maybe_set(intr, "fov_h_deg", "ptz_intrinsics_fov_h_deg", float)
                        ptz["intrinsics"] = intr

                        # Zoom behavior (auto-track distance control)
                        _maybe_set(ptz, "zoom_target_m",         "ptz_zoom_target_m", float)
                        _maybe_set(ptz, "zoom_deadband_m",       "ptz_zoom_deadband_m", float)
                        _maybe_set(ptz, "zoom_kp",               "ptz_zoom_kp", float)
                        _maybe_set(ptz, "zoom_seconds",          "ptz_zoom_seconds", float)
                        _maybe_set(ptz, "zoom_enable_error_deg", "ptz_zoom_enable_error_deg", float)

                        # Autotrack toggle
                        if "ptz_autotrack" in request.form:
                            config["ptz_autotrack"] = bool(request.form.get("ptz_autotrack"))

                        # Write back
                        config["ptz"] = ptz

                    except Exception as _e:
                        logger.warning(f"[CONTROL] PTZ config parse skipped: {_e}")

                    if save_config(config, write_reload_flag=True):
                        try:
                            globals()["config"] = load_config()
                        except Exception:
                            pass
                        message = "Configuration updated successfully."
                    else:
                        message = "Failed to save configuration."
                        
                elif action == "validate_snapshots":
                    invalid_count = validate_snapshots()
                    message = f"Snapshot validation complete. {invalid_count} invalid paths cleaned."
                        
            except Exception as e:
                logger.error(f"Control action error: {e}")
                message = f"Action failed: {str(e)}"
        
        # Get system stats
        try:
            disk_usage = psutil.disk_usage('/')
            disk_free = disk_usage.free / (1024**3)
        except Exception:
            disk_free = 0

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("SELECT COUNT(*) FROM radar_data")
                total_records = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM radar_data WHERE snapshot_path IS NOT NULL")
                snapshot_records = cursor.fetchone()[0]
        except Exception:
            total_records = 0
            snapshot_records = 0

        # --- Radar health (prefer Pi health HTTP; fallback to local UART) ---
        radar_ok = False
        # If PI_HEALTH_URL is configured (env or config), query it for radar_connected
        try:
            health_url = None
            if PI_HEALTH_URL:
                health_url = PI_HEALTH_URL.strip()
                if not health_url.endswith("/health"):
                    health_url = health_url.rstrip("/") + "/health"
            if health_url:
                r = requests.get(health_url, timeout=2.5)
                if r.status_code == 200:
                    j = r.json()
                    radar_ok = bool(j.get("radar_connected", False))
                else:
                    logger.warning(f"[RADAR HEALTH] Pi health returned {r.status_code}")
        except Exception as e:
            logger.warning(f"[RADAR HEALTH] Pi health probe failed: {e}")

        # Fallback to legacy UART poke only if HTTP health is not available / false
        if not radar_ok:
            cli_ser = data_ser = None
            try:
                ports = (config.get("iwr_ports") or {})
                cli = ports.get("cli")
                data = ports.get("data")
                cli_ser  = check_radar_connection(cli, 115200, timeout=0.6) if cli else None
                data_ser = check_radar_connection(data, 921600, timeout=0.6) if data else None
                radar_ok = bool(cli_ser) and bool(data_ser)
            except Exception as e:
                logger.warning(f"[RADAR TEST] Interface error: {e}")
                radar_ok = False
            finally:
                try:
                    if cli_ser: cli_ser.close()
                except: pass
                try:
                    if data_ser: data_ser.close()
                except: pass

        cams = config.get("cameras", [])
        selected = config.get("selected_camera", 0)
        cam = cams[selected] if cams and selected < len(cams) else {}
        stream_type = cam.get("stream_type", "mjpeg")
        camera_ok = False

        should_check_camera = action != "test_camera" if request.method == "POST" else True

        if should_check_camera:
            try:
                url = cam.get("url", "")
                username = cam.get("username", "")
                password = cam.get("password", "")
                if stream_type == "rtsp":
                    if url.startswith("rtsp://") and "@" not in url and username and password:
                        url = url.replace("rtsp://", f"rtsp://{username}:{password}@")

                    logger.info(f"[CONTROL CAMERA TEST] RTSP URL: {url}")
                    result = subprocess.run(
                        ["ffmpeg", "-rtsp_transport", "tcp", "-i", url, "-t", "1", "-f", "null", "-"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    camera_ok = (result.returncode == 0)

                elif stream_type == "mjpeg":
                    r = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True, timeout=5)
                    if r.status_code == 200:
                        buffer = b""
                        for chunk in r.iter_content(1024):
                            buffer += chunk
                            if b'\xff\xd8' in buffer and b'\xff\xd9' in buffer:
                                camera_ok = True
                                break
                    r.close()

                elif stream_type == "snapshot":
                    r = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=5)
                    if r.status_code == 200 and r.content.startswith(b'\xff\xd8'):
                        camera_ok = True

                logger.info(f"[CONTROL CAMERA TEST RESULT] camera_ok = {camera_ok}")

            except Exception as e:
                logger.warning(f"[CONTROL CAMERA TEST] Unexpected failure: {e}")
        basedir = os.path.dirname(os.path.abspath(__file__))
        try:
            db_cameras, _sel_idx = load_cameras_from_db()
            active_cameras = [c for c in db_cameras if c.get("enabled", True)]
            return render_template("control.html",
                message=message,
                config=config,
                db_cameras=db_cameras,
                active_cameras=active_cameras,
                disk_free=round(disk_free, 2),
                total_records=total_records,
                snapshot_records=snapshot_records,
                snapshot=snapshot,
                projection_matrix_exists = os.path.exists(os.path.join(basedir, "calibration", "camera_projection_matrix.npy")),
                radar_status=radar_ok,
                camera_status=camera_ok,
                model_info=get_model_metadata()
            )
        except Exception as e:
            import traceback
            logger.error(f"[CONTROL PAGE ERROR] {e}\n{traceback.format_exc()}")
            return f"<pre>{traceback.format_exc()}</pre>", 500
    
    @app.route("/users", methods=["GET", "POST"])
    @login_required
    def users():
        if request.method == "POST":
            action = request.form.get("action")
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                    if action == "add_user":
                        if not is_admin():
                            flash("Admin access required", "error")
                            return redirect(url_for("users"))

                        username = request.form.get("username", "").strip()
                        password = request.form.get("password", "").strip()
                        role = request.form.get("role", "viewer")

                        if not username or not password:
                            flash("Username and password are required", "error")
                        elif len(password) < 6:
                            flash("Password must be at least 6 characters", "error")
                        elif role not in ["admin", "viewer"]:
                            flash("Invalid role", "error")
                        else:
                            cursor.execute("""
                                INSERT INTO users (username, password_hash, role)
                                VALUES (%s, %s, %s)
                            """, (username, generate_password_hash(password), role))
                            conn.commit()
                            flash(f"User '{username}' added successfully.", "success")

                    elif action == "change_password":
                        current_password = request.form.get("current_password", "")
                        new_password = request.form.get("new_password", "")
                        confirm_password = request.form.get("confirm_password", "")
                        user = get_user_by_id(current_user.id)

                        if not all([current_password, new_password, confirm_password]):
                            flash("All password fields are required", "error")
                        elif new_password != confirm_password:
                            flash("New passwords do not match", "error")
                        elif len(new_password) < 6:
                            flash("New password must be at least 6 characters", "error")
                        elif not user or not check_password_hash(user.password_hash, current_password):
                            flash("Current password is incorrect", "error")
                        else:
                            cursor.execute("""
                                UPDATE users SET password_hash = %s WHERE id = %s
                            """, (generate_password_hash(new_password), user.id))
                            conn.commit()
                            flash("Password changed successfully", "success")

            except psycopg2.IntegrityError:
                flash("Username already exists.", "error")
            except Exception as e:
                logger.error(f"[USER MANAGEMENT ERROR] {e}")
                flash("An error occurred while managing users.", "error")

            return redirect(url_for("users"))

        # --- GET: list users; render IST times
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT
                            u.id,
                            u.username,
                            u.role,
                            u.created_at,
                            ua.last_activity
                        FROM users u
                        LEFT JOIN user_activity ua ON ua.user_id = u.id
                        ORDER BY u.id;
                    """)
                    raw = [dict(r) for r in cur.fetchall()]

            rows = []
            now_utc = datetime.now(timezone.utc)
            for r in raw:
                # last_activity may be naive; treat as UTC
                last = r.get("last_activity")
                if isinstance(last, str):
                    try:
                        last = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    except Exception:
                        last = None
                if last is not None and getattr(last, "tzinfo", None) is None:
                    last = last.replace(tzinfo=timezone.utc)

                is_active = False
                if last is not None:
                    is_active = (now_utc - last) <= timedelta(minutes=30)

                r["created_at_ist"]    = to_ist(r.get("created_at"))
                r["last_activity_ist"] = to_ist(last)
                r["active"] = is_active
                r["is_active"] = is_active
                rows.append(r)

            return render_template("users.html", users=rows, users_list=rows, total=len(rows))
        except Exception as e:
            logger.exception("[USERS LOAD ERROR] %s", e)
            flash("Error loading user list", "error")
            return render_template("users.html", users=[], users_list=[], total=0)
    
    @app.route('/delete_user/<int:user_id>', methods=['POST'])
    @login_required
    def delete_user(user_id):
        if not current_user.is_authenticated:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({'success': False, 'error': 'Session expired'}), 401
            return redirect(url_for('login'))

        if current_user.role != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        if current_user.id == user_id:
            return jsonify({'success': False, 'error': 'You cannot delete your own account'}), 400

        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()

                if not user:
                    return jsonify({'success': False, 'error': 'User not found'}), 404

                cursor.execute("DELETE FROM user_activity WHERE user_id = %s", (user_id,))
                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
                conn.commit()

            logger.info(f"[DELETE USER] Deleted user ID {user_id}")
            return jsonify({'success': True}), 200

        except Exception as e:
            logger.exception(f"[DELETE USER ERROR] {e}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route("/change_password", methods=["POST"])
    @login_required
    def change_password():
        try:
            current_password = request.form.get("current_password", "").strip()
            new_password = request.form.get("new_password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if not all([current_password, new_password, confirm_password]):
                flash("All fields are required", "error")
                return redirect(url_for("users"))

            if new_password != confirm_password:
                flash("New passwords do not match", "error")
                return redirect(url_for("users"))

            if len(new_password) < 6:
                flash("New password must be at least 6 characters", "error")
                return redirect(url_for("users"))

            user = get_user_by_id(current_user.id)
            if not user or not check_password_hash(user.password_hash, current_password):
                flash("Current password is incorrect", "error")
                return redirect(url_for("users"))

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute("""
                    UPDATE users SET password_hash = %s WHERE id = %s
                """, (generate_password_hash(new_password), user.id))
                conn.commit()

            flash("Password changed successfully", "success")
            logger.info(f"[PASSWORD CHANGE] User {current_user.username} changed password.")
            return redirect(url_for("users"))

        except Exception as e:
            logger.error(f"[PASSWORD CHANGE ERROR] {e}")
            flash("Error changing password", "error")
            return redirect(url_for("users"))
    
    @app.route("/api/active_users")
    @login_required
    def api_active_users():
        try:
            active_users = get_active_users(minutes=30)  
            return jsonify({
                "active_count": len(active_users),
                "active_users": [
                    {
                        "username": user['username'],
                        "role": user['role'],
                        "last_activity": user['last_activity']
                    } for user in active_users
                ]
            })
        except Exception as e:
            logger.error(f"[API ACTIVE USERS ERROR] {e}")
            return jsonify({"error": "Internal server error"}), 500
        
    return app

ensure_directories()
flask_app = create_app()

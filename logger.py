# logger.py — daily rotation, 3-day retention, IST timestamps, backward compatible
import logging
import os
import gzip
import shutil
import threading
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone, timedelta
import time
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ── settings ──────────────────────────────────────────────────────────────────
LOG_DIR   = os.environ.get("LOG_DIR", "system-logs")
LOG_FILE  = os.environ.get("LOG_FILE", os.path.join(LOG_DIR, "radar.log"))  # original default
KEEP_DAYS = int(os.environ.get("LOG_KEEP_DAYS", "3"))  # uniform retention across all python logs
WHEN, UTC = "midnight", False  # local midnight rotation

# timezone for formatting (display). Default IST; override via env LOG_TZ if needed.
LOG_TZ = os.environ.get("LOG_TZ", "Asia/Kolkata")

os.makedirs(LOG_DIR, exist_ok=True)

# ── timezone-aware formatter (renders in IST by default) ──────────────────────
class TZFormatter(logging.Formatter):
    """
    Formats record times in a specific timezone (default Asia/Kolkata).
    Uses zoneinfo if available; falls back to fixed +05:30 offset.
    """
    def __init__(self, *args, tz_name: str = LOG_TZ, **kwargs):
        super().__init__(*args, **kwargs)
        self._tz = ZoneInfo(tz_name) if ZoneInfo else timezone(timedelta(hours=5, minutes=30))

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self._tz)
        if datefmt:
            return dt.strftime(datefmt)
        # default like logging.Formatter but with milliseconds
        s = dt.strftime("%Y-%m-%d %H:%M:%S")
        return f"{s},{int(record.msecs):03d}"

# ── thread-safe console handler ───────────────────────────────────────────────
class SafeConsoleHandler(logging.StreamHandler):
    _lock = threading.Lock()
    def emit(self, record):
        with self._lock:
            try:
                super().emit(record)
            except Exception:
                # never raise from logging
                pass

# ── daily rotating file handler with gzip compression ─────────────────────────
class SafeTimedRotatingHandler(TimedRotatingFileHandler):
    """
    Rolls daily at local midnight, keeps backupCount days, gzips rolled files.
    Thread-safe emit to avoid interleaving from multiple threads.
    """
    _emit_lock = threading.Lock()

    def __init__(self, filename, when=WHEN, interval=1, backupCount=KEEP_DAYS, utc=UTC, encoding="utf-8"):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        super().__init__(filename, when=when, interval=interval, backupCount=backupCount, utc=utc, encoding=encoding)

    def rotation_filename(self, default_name: str) -> str:
        # default_name like: system-logs/radar.log.2025-10-30
        return default_name  # we'll add .gz in rotate()

    def rotate(self, source: str, dest: str) -> None:
        # gzip the rotated file; fall back to rename if compression fails
        try:
            with open(source, "rb") as f_in, gzip.open(dest + ".gz", "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
            try:
                os.remove(source)
            except Exception:
                pass
        except Exception:
            try:
                os.replace(source, dest)
            except Exception:
                pass

    def emit(self, record):
        with self._emit_lock:
            try:
                super().emit(record)
            except Exception:
                pass

# ── public API ────────────────────────────────────────────────────────────────
def setup_logger(
    name: str = "radar_logger",
    log_file: str = LOG_FILE,
    level: int = logging.INFO,
    to_console: bool = True,
) -> logging.Logger:
    """
    Create (or return existing) logger that writes to a daily-rotating, gzipped file.
    Display timestamps are in IST (Asia/Kolkata) by default.
    Retention is KEEP_DAYS (default 3) uniformly.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_isk_logger_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)  # handler-level filtering will apply

    # file handler
    fh = SafeTimedRotatingHandler(
        filename=log_file,
        when=WHEN,
        interval=1,
        backupCount=KEEP_DAYS,
        utc=UTC,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(TZFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logger.addHandler(fh)

    # optional console echo (NSSM/stdout collectors)
    if to_console:
        ch = SafeConsoleHandler()
        ch.setLevel(level)
        ch.setFormatter(TZFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)

    logger._isk_logger_configured = True  # type: ignore[attr-defined]
    return logger

def get_logger(child_name: str = None) -> logging.Logger:
    """
    Returns the base logger or a namespaced child (e.g., get_logger('pipeline')).
    Ensures base is initialized with DEFAULT_LOG_FILE and IST formatting.
    """
    base = setup_logger()
    return base.getChild(child_name) if child_name else base

def new_file_logger(name: str, log_file: str, level: int = logging.INFO, to_console: bool = True) -> logging.Logger:
    """
    Create a separate named logger that writes to its own file with the same rotation,
    retention policy, and IST timestamp formatting.
    """
    lg = logging.getLogger(name)
    if getattr(lg, "_isk_logger_configured", False):
        return lg

    lg.setLevel(logging.DEBUG)

    fh = SafeTimedRotatingHandler(
        filename=log_file,
        when=WHEN,
        interval=1,
        backupCount=KEEP_DAYS,
        utc=UTC,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(TZFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    lg.addHandler(fh)

    if to_console:
        ch = SafeConsoleHandler()
        ch.setLevel(level)
        ch.setFormatter(TZFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        lg.addHandler(ch)

    lg._isk_logger_configured = True  # type: ignore[attr-defined]
    return lg

logger = setup_logger()
import logging

def _configure_logger():
    lg = logging.getLogger("isk")
    if lg.handlers:  # idempotent
        return lg
    lg.setLevel(logging.INFO)
    h = logging.FileHandler("system-logs/isk-app.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    lg.addHandler(h)
    return lg

logger = _configure_logger()

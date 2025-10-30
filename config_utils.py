import json
import os
from logger import logger
from urllib.parse import urlparse

CONFIG_FILE = os.environ.get("APP_CONFIG", "app_config.json")

def _deep_merge(dst: dict, src: dict) -> dict:
    """
    Recursive merge that preserves existing keys in dst
    and fills only missing ones from src. Dicts merge; scalars
    only set if absent.
    """
    out = dict(dst or {})
    for k, v in (src or {}).items():
        if isinstance(v, dict):
            out[k] = _deep_merge(out.get(k, {}), v)
        else:
            if k not in out:
                out[k] = v
    return out

_DEFAULTS = {
    # Core
    "cooldown_seconds": 0.5,
    "retention_days": 30,
    "selected_camera": 0,
    "annotation_conf_threshold": 0.5,
    "label_format": "{type} | {speed:.1f} km/h",
    "acceleration_threshold": 2.0,
    "min_acc_violation_frames": 3,
    "absolute_max_speed_kmh": 220.0,
    "absolute_max_distance_m": 200.0,
    "tracker_max_range_m": 150.0,
    "tracker_min_range_m": 0.20,
    "tracker_max_radial_mps": 80.0,
    "tracker_max_speed_mps": 80.0,
    "tracker_low_snr_margin_db": 3.0,
    "ptz_autotrack": False,

    # PTZ defaults 
    "ptz": {
        "host": "",
        "username": "",
        "password": "",
        "port": 80,
        "scheme": "http",
        "auth_mode": "auto",
        # legacy UI params 
        "kp": 0.015,
        "deadband_deg": 1.5,
        "nudge_seconds": 0.35,
        "max_rate_hz": 3.0,
        "ui_velocity": 0.40,
        "ui_seconds": 0.60,
        # structured blocks
        "mount": {                # radar->PTZ transform (PTZ frame origin at PTZ pivot)
            "dx_m": 0.0,          # +X right
            "dy_m": 0.0,          # +Y forward
            "dz_m": 0.0,          # +Z up (negative if PTZ is below radar)
            "yaw_deg": 0.0,       # rotation about Z
            "pitch_deg": 0.0,     # rotation about X
            "roll_deg": 0.0       # rotation about Y
        },
        "home": {                 # deterministic start pose
            "pan_deg": 0.0,
            "tilt_deg": -2.0,
            "zoom": 1.5,
            "wait_settle": True,
            "settle_s": 0.8
        },
        "control": {              # continuous move controller
            "deadband_deg": 0.35,
            "k_pan": 0.70,
            "k_tilt": 0.60,
            "max_vx": 0.85,
            "max_vy": 0.85,
            "max_step_s": 0.25,
            "stop_pulse_s": 0.08
        },
        "zoom": { "min": 1.0, "max": 20.0 },
        "intrinsics": { "fov_h_deg": 60.0 },
        "auto_track": { "home_on_enable": True, "settle_s": 1.20, "acquire_timeout_s": 4.0 },
        "zoom_target_m": 12.0,
        "zoom_deadband_m": 2.0,
        "zoom_kp": 0.08,
        "zoom_seconds": 0.35,
        "zoom_enable_error_deg": 4.0
    },

    # IWR ports
    "iwr_ports": {
        "cli": "/dev/ttyACM1",
        "data": "/dev/ttyACM2",
        "cfg_path": "isk_config.cfg"
    },

    # Network radar defaults (PC consumes, Pi serves)
    "radar_over_ip": {
        "enabled": False,
        "role": "pc",              # "pc" or "pi-bridge"
        "host": "",
        "bind": "0.0.0.0",
        "port": 6868,
        "format": "jsonl",
        "compress": False,
        "heartbeat_seconds": 2.0,
        "stale_cutoff_seconds": 5.0
    },
    # Database defaults (used mainly on PC)
    "db": {
        "dsn": "",
        "host": "localhost",
        "user": "radar_user",
        "password": "",
        "name": "iwr6843_db",
        "sslmode": "disable"
    },

    # Cameras (list)
    "cameras": [{
        "url": "",
        "snapshot_url": "",
        "username": "",
        "password": "",
        "stream_type": "mjpeg",
        "enabled": True
    }],

    # Speed limits
    "dynamic_speed_limits": {
        "default": 3.0,
        "HUMAN": 4.0,
        "CAR": 70.0,
        "TRUCK": 50.0,
        "BUS": 50.0,
        "BIKE": 60.0,
        "BICYCLE": 10.0,
        "UNKNOWN": 50.0
    },

    # Projection / calibration UI expects these:
    "use_speed_correction": True,
    "calibration_mode": False,
    "camera_width_px": 1280,
    "camera_height_px": 720,
    "camera_fov_h_deg": 90.0,
    # k1..k5 (OpenCV) — keep 5 entries
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "calibration_min_points": 12,
    "calibration_reproj_ok_px": 6.0,

    # Vision motion gate
    "vision_motion_threshold": 3.0,

    # Watchdog / radar
    "radar_get_timeout": 0.8,
    "radar_stall_retries": 3,
    "watchdog_hard_exit_after": 0,

    # Kafka (optional)
    "kafka": {
        "enabled": False,
        "bootstrap_servers": "localhost:9092",
        "client_id": "iwr6843-app",
        "acks": 1,
        "compression_type": "gzip",
        "topic_violation": "violations",
        "topic_metrics": "metrics",
        "topic_events": "events",
        "topic_frames": "camera_frames",
        "topic_clips": "camera_clips",
        "topic_plotter": "plotter_frames",
        "ssl": {
            "enabled": False,
            "cafile": "",
            "certfile": "",
            "keyfile": "",
            "check_hostname": True
        }
    }
}

def _normalize(cfg: dict) -> dict:
    """Fill missing keys and coerce types so the rest of the app never crashes."""
    cfg = dict(cfg or {})

    # Fill simple defaults
    for k, v in _DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v

    # Nested: cameras
    cams = cfg.get("cameras")
    if not isinstance(cams, list) or not all(isinstance(c, dict) for c in cams):
        cfg["cameras"] = list(_DEFAULTS["cameras"])
    # ensure selected_camera is in range at runtime

    # Nested: iwr_ports
    if not isinstance(cfg.get("iwr_ports"), dict):
        cfg["iwr_ports"] = dict(_DEFAULTS["iwr_ports"])

    # Nested: radar_over_ip
    if not isinstance(cfg.get("radar_over_ip"), dict):
        cfg["radar_over_ip"] = dict(_DEFAULTS["radar_over_ip"])
    else:
        cfg["radar_over_ip"] = _deep_merge(cfg["radar_over_ip"], _DEFAULTS["radar_over_ip"])
    try:
        role = str(cfg["radar_over_ip"].get("role", "pc")).lower()
        if role not in ("pc", "pi-bridge"):
            role = "pc"
        cfg["radar_over_ip"]["role"] = role
    except Exception:
        cfg["radar_over_ip"]["role"] = "pc"

    # Nested: db
    if not isinstance(cfg.get("db"), dict):
        cfg["db"] = dict(_DEFAULTS["db"])
    else:
        cfg["db"] = _deep_merge(cfg["db"], _DEFAULTS["db"])
    # env → DB DSN override
    try:
        dsn_env = os.environ.get("DB_DSN") or os.environ.get("DATABASE_URL")
        if dsn_env:
            cfg["db"]["dsn"] = dsn_env
    except Exception:
        pass

    # Nested: speed limits
    if not isinstance(cfg.get("dynamic_speed_limits"), dict):
        cfg["dynamic_speed_limits"] = dict(_DEFAULTS["dynamic_speed_limits"])
    if "default" not in cfg["dynamic_speed_limits"]:
        cfg["dynamic_speed_limits"]["default"] = _DEFAULTS["dynamic_speed_limits"]["default"]

    if not isinstance(cfg.get("ptz"), dict):
        cfg["ptz"] = dict(_DEFAULTS["ptz"])
    else:
        cfg["ptz"] = _deep_merge(cfg["ptz"], _DEFAULTS["ptz"])
    cfg["ptz_autotrack"] = bool(cfg.get("ptz_autotrack", _DEFAULTS["ptz_autotrack"]))

    try:
        if "deadband_deg" in cfg["ptz"] and "deadband_deg" not in cfg["ptz"]["control"]:
            cfg["ptz"]["control"]["deadband_deg"] = float(cfg["ptz"]["deadband_deg"])
    except Exception:
        pass

    # Coerce PTZ numeric fields
    def _coerce_float(d: dict, keys: list):
        for k in keys:
            if k in d:
                try: d[k] = float(d[k])
                except Exception: pass
    def _coerce_int(d: dict, keys: list):
        for k in keys:
            if k in d:
                try: d[k] = int(d[k])
                except Exception: pass

    try:
        _coerce_int(cfg["ptz"], ["port"])
        # mount
        _coerce_float(cfg["ptz"]["mount"], ["dx_m","dy_m","dz_m","yaw_deg","pitch_deg","roll_deg"])
        # home
        _coerce_float(cfg["ptz"]["home"], ["pan_deg","tilt_deg","zoom"])
        # control
        _coerce_float(cfg["ptz"]["control"], ["deadband_deg","k_pan","k_tilt","max_vx","max_vy","max_step_s","stop_pulse_s"])
        # zoom
        _coerce_float(cfg["ptz"]["zoom"], ["min","max"])
        # intrinsics
        _coerce_float(cfg["ptz"]["intrinsics"], ["fov_h_deg"])
        # auto_track
        _coerce_float(cfg["ptz"]["auto_track"], ["settle_s","acquire_timeout_s"])
        if "home_on_enable" in cfg["ptz"]["auto_track"]:
            cfg["ptz"]["auto_track"]["home_on_enable"] = bool(cfg["ptz"]["auto_track"]["home_on_enable"])
        _coerce_float(cfg["ptz"], ["zoom_target_m","zoom_deadband_m","zoom_kp","zoom_seconds","zoom_enable_error_deg"])
    except Exception:
        pass

    try:
        _coerce_int(cfg["radar_over_ip"], ["port"])
        _coerce_float(cfg["radar_over_ip"], ["heartbeat_seconds","stale_cutoff_seconds"])
    except Exception:
        pass

    # Nested: kafka
    try:
        if not isinstance(cfg.get("kafka"), dict):
            cfg["kafka"] = dict(_DEFAULTS["kafka"])  # shallow copy ok
        else:
            cfg["kafka"] = _deep_merge(cfg["kafka"], _DEFAULTS["kafka"])
        # coerce/normalize expected types
        cfg["kafka"]["enabled"] = bool(cfg["kafka"].get("enabled", False))
        cfg["kafka"]["acks"] = int(cfg["kafka"].get("acks", 1))
        cfg["kafka"]["compression_type"] = str(cfg["kafka"].get("compression_type") or "") or None
        if not isinstance(cfg["kafka"].get("ssl"), dict):
            cfg["kafka"]["ssl"] = dict(_DEFAULTS["kafka"]["ssl"])  # shallow copy ok
        else:
            cfg["kafka"]["ssl"] = _deep_merge(cfg["kafka"]["ssl"], _DEFAULTS["kafka"]["ssl"])
        cfg["kafka"]["ssl"]["enabled"] = bool(cfg["kafka"]["ssl"].get("enabled", False))
        cfg["kafka"]["ssl"]["check_hostname"] = bool(cfg["kafka"]["ssl"].get("check_hostname", True))
    except Exception:
        # If kafka block is malformed, fall back to defaults silently
        cfg["kafka"] = dict(_DEFAULTS["kafka"])  

    # Distortion: always 5 floats
    dist = cfg.get("distortion", _DEFAULTS["distortion"])
    if not isinstance(dist, list):
        dist = _DEFAULTS["distortion"]
    dist = [float(x) for x in (dist + [0, 0, 0, 0, 0])[:5]]
    cfg["distortion"] = dist

    # Types for intrinsics
    try:
        cfg["camera_width_px"]  = int(cfg.get("camera_width_px",  _DEFAULTS["camera_width_px"]))
        cfg["camera_height_px"] = int(cfg.get("camera_height_px", _DEFAULTS["camera_height_px"]))
        cfg["camera_fov_h_deg"] = float(cfg.get("camera_fov_h_deg", _DEFAULTS["camera_fov_h_deg"]))
    except Exception:
        cfg["camera_width_px"]  = _DEFAULTS["camera_width_px"]
        cfg["camera_height_px"] = _DEFAULTS["camera_height_px"]
        cfg["camera_fov_h_deg"] = _DEFAULTS["camera_fov_h_deg"]

    # Booleans
    cfg["use_speed_correction"] = bool(cfg.get("use_speed_correction", True))
    cfg["calibration_mode"]     = bool(cfg.get("calibration_mode", False))

    # Physics caps must be floats (JSON may contain strings)
    try:
        cfg["absolute_max_speed_kmh"] = float(
            cfg.get("absolute_max_speed_kmh", _DEFAULTS["absolute_max_speed_kmh"])
        )
    except Exception:
        cfg["absolute_max_speed_kmh"] = _DEFAULTS["absolute_max_speed_kmh"]
    try:
        cfg["absolute_max_distance_m"] = float(
            cfg.get("absolute_max_distance_m", _DEFAULTS["absolute_max_distance_m"])
        )
    except Exception:
        cfg["absolute_max_distance_m"] = _DEFAULTS["absolute_max_distance_m"]

    def _f(key):
        try:
            return float(cfg.get(key, _DEFAULTS[key]))
        except Exception:
            return float(_DEFAULTS[key])
    cfg["tracker_max_range_m"]       = _f("tracker_max_range_m")
    cfg["tracker_min_range_m"]       = _f("tracker_min_range_m")
    cfg["tracker_max_radial_mps"]    = _f("tracker_max_radial_mps")
    cfg["tracker_max_speed_mps"]     = _f("tracker_max_speed_mps")
    cfg["tracker_low_snr_margin_db"] = _f("tracker_low_snr_margin_db")

    try:
        if not cfg["ptz"].get("host"):
            cams = cfg.get("cameras", [])
            for cam in cams:
                role = str(cam.get("role","")).lower()
                if role != "ptz":
                    continue
                host = cam.get("host") or cam.get("ip")
                port = cam.get("port")
                user = cam.get("username")
                pwd  = cam.get("password")
                # Parse from URL if needed
                if (not host) and cam.get("url"):
                    u = urlparse(str(cam["url"]))
                    if u.hostname: host = u.hostname
                    if u.port:     port = u.port
                    # If URL embeds creds and explicit fields are empty, use them
                    if (not user) and u.username: user = u.username
                    if (not pwd)  and u.password: pwd  = u.password
                if host:
                    cfg["ptz"]["host"] = host
                    if port: cfg["ptz"]["port"] = int(port)
                    if user: cfg["ptz"]["username"] = user
                    if pwd:  cfg["ptz"]["password"] = pwd
                    break
    except Exception:
        # keep defaults if anything goes wrong
        pass

    return cfg

def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r") as f:
            existing = json.load(f)
        return _normalize(existing)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("[CONFIG] Using defaults (no/invalid config file)")
        # Don’t crash if file is missing; return normalized defaults.
        return _normalize({})

def save_config(cfg: dict, write_reload_flag: bool = False) -> bool:
    """
    Persist config and (optionally) touch reload flag for the running pipeline.
    """
    try:
        cfg = _normalize(cfg)
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        if write_reload_flag:
            try:
                open("reload_flag.txt", "w").close()
            except Exception:
                pass
        return True
    except Exception as e:
        logger.error(f"[CONFIG] save failed: {e}")
        return False

def db_dsn(cfg: dict) -> str:
    """
    Return a DSN string. If cfg['db']['dsn'] is empty, build it from parts.
    Env DB_DSN/DATABASE_URL already handled in _normalize().
    """
    cfg = _normalize(cfg)
    db = cfg.get("db", {})
    dsn = (db.get("dsn") or "").strip()
    if dsn:
        return dsn
    host = db.get("host", "localhost")
    name = db.get("name", "iwr6843_db")
    user = db.get("user", "radar_user")
    pwd  = db.get("password", "essi")
    ssl  = db.get("sslmode", "disable")
    return f"dbname={name} user={user} password={pwd} host={host} sslmode={ssl}"

def radar_role(cfg: dict) -> str:
    """Return 'pc' or 'pi-bridge'."""
    try:
        return str(cfg.get("radar_over_ip", {}).get("role", "pc")).lower()
    except Exception:
        return "pc"

def is_pc_consumer(cfg: dict) -> bool:
    r = cfg.get("radar_over_ip", {})
    return bool(r.get("enabled")) and str(r.get("role","pc")).lower() == "pc"

def is_pi_bridge(cfg: dict) -> bool:
    r = cfg.get("radar_over_ip", {})
    return bool(r.get("enabled")) and str(r.get("role","")).lower() == "pi-bridge"
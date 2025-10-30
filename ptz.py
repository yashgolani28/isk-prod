import argparse, threading, time, requests, sys, contextlib, math
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import tkinter as tk
from tkinter import ttk, messagebox

# ---------- iAPI client ----------
class PTZClient:
    def __init__(self, host, user, pwd, scheme="http", port=None, auth_mode="auto", insecure=False, timeout=0.7):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.scheme = scheme
        self.port = port if port is not None else (443 if scheme == "https" else 80)
        self.auth_mode = auth_mode
        self.verify = False if insecure else True
        self.timeout = timeout
        self.sess = requests.Session()
        self._lock = threading.Lock()

    def _url(self, path):
        return f"{self.scheme}://{self.host}:{self.port}/iAPI/{path}"

    def _get(self, path, params):
        url = self._url(path)
        with self._lock:
            if self.user and self.pwd:
                if self.auth_mode == "basic":
                    r = self.sess.get(url, params=params, auth=HTTPBasicAuth(self.user, self.pwd),
                                      timeout=self.timeout, verify=self.verify)
                elif self.auth_mode == "digest":
                    r = self.sess.get(url, params=params, auth=HTTPDigestAuth(self.user, self.pwd),
                                      timeout=self.timeout, verify=self.verify)
                else:
                    # auto: try digest then basic
                    r = self.sess.get(url, params=params, auth=HTTPDigestAuth(self.user, self.pwd),
                                      timeout=self.timeout, verify=self.verify)
                    if r.status_code in (401, 403):
                        r = self.sess.get(url, params=params, auth=HTTPBasicAuth(self.user, self.pwd),
                                          timeout=self.timeout, verify=self.verify)
            else:
                r = self.sess.get(url, params=params, timeout=self.timeout, verify=self.verify)
        r.raise_for_status()
        return r.text

    def status(self):
        return self._get("ptzfi.cgi", {"action": "Status"})

    def status_parsed(self) -> dict:
        """
        Return a dict parsed from ptzfi.cgi?action=Status.
        Typical keys: pan, tilt, zoom, pansp, tiltsp, zoomsp, autofocus, autoiris, autotrackmode, etc.
        """
        out = {}
        try:
            txt = self.status()
            for line in txt.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
        except Exception:
            pass
        # Coerce some numeric fields when present
        for k in ("pan","tilt","zoom","pansp","tiltsp","zoomsp","focus","focussp","irisevsp"):
            if k in out:
                with contextlib.suppress(Exception):
                    out[k] = float(out[k])
        return out

    def explain(self):
        return self._get("ptzfi.cgi", {"action": "Explain"})

    def stop(self):
        return self._get("ptzfi.cgi", {"action": "Stop"})

    def stop_hard(self):
        """
        Extra-stable stop to prevent residual drift on some firmwares:
        Stop → send zero speeds → tiny wait → Stop again.
        """
        with self._lock:
            # best-effort; ignore individual failures
            with contextlib.suppress(Exception):
                self.sess.get(self._url("ptzfi.cgi"), params={"action": "Stop"},
                              timeout=self.timeout, verify=self.verify)
            with contextlib.suppress(Exception):
                self.sess.get(self._url("ptzfi.cgi"),
                              params={"action": "ContinuousMove", "pansp": 0, "tiltsp": 0, "zoomsp": 0},
                              timeout=self.timeout, verify=self.verify)
            time.sleep(0.05)
            with contextlib.suppress(Exception):
                self.sess.get(self._url("ptzfi.cgi"), params={"action": "Stop"},
                              timeout=self.timeout, verify=self.verify)

    def continuous(self, pansp=0, tiltsp=0, zoomsp=0):
        pansp  = max(-100, min(100, int(pansp)))
        tiltsp = max(-100, min(100, int(tiltsp)))
        zoomsp = max(-100, min(100, int(zoomsp)))
        return self._get("ptzfi.cgi", {"action": "ContinuousMove", "pansp": pansp, "tiltsp": tiltsp, "zoomsp": zoomsp})

    def relative(self, dpan=0.0, dtilt=0.0, dzoom=0.0):
        return self._get("ptzfi.cgi", {"action": "RelativeMove", "units": "degrees",
                                       "pan": dpan, "tilt": dtilt, "zoom": dzoom})

    def absolute(self, pan=None, tilt=None, zoom=None):
        q = {"action": "AbsoluteMove"}
        if pan  is not None: q["pan"]  = float(pan)
        if tilt is not None: q["tilt"] = float(tilt)
        if zoom is not None: q["zoom"] = float(zoom)
        return self._get("ptzfi.cgi", q)

    def presets_read(self):
        return self._get("presets.cgi", {"action": "Read"})

    def preset_create(self, name):
        return self._get("presets.cgi", {"action": "Create", "name": name})

    def preset_goto(self, token):
        # accept name or id; resolve if needed
        t = str(token)
        if not t.isdigit():
            listing = self.presets_read()
            pid = self._find_preset_id_by_name(listing, t)
            if pid is None:
                raise RuntimeError(f"Preset not found: {t}")
            t = str(pid)
        return self._get("presets.cgi", {"action": "Goto", "id": t})

    def preset_delete(self, token):
        t = str(token)
        if not t.isdigit():
            listing = self.presets_read()
            pid = self._find_preset_id_by_name(listing, t)
            if pid is None:
                raise RuntimeError(f"Preset not found: {t}")
            t = str(pid)
        return self._get("presets.cgi", {"action": "Delete", "id": t})

    @staticmethod
    def _find_preset_id_by_name(listing_text, name):
        for line in listing_text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].startswith("id=") and parts[1].startswith("name="):
                pid = parts[0][3:]
                pname = parts[1][5:]
                if pname == name:
                    try:
                        return int(pid)
                    except ValueError:
                        pass
        return None

    def close(self):
        with contextlib.suppress(Exception):
            self.sess.close()

    def __del__(self):
        self.close()

# ---------- Programmatic controller ----------
class PTZError(Exception):
    pass

class PTZController:
    """
    Thin wrapper over PTZClient exposing:
      - continuous_move(vx, vy, duration): normalized speeds in [-1..+1]
      - stop()
      - goto_preset(name_or_id)
    """
    def __init__(self, host, username="", password="", scheme="http", port=None,
                 auth_mode="auto", insecure=False, timeout=0.7,
                 invert_pan=False, invert_tilt=False,
                 profile_token: str | None = None,
                 max_pan_speed: float = 0.70,
                 max_tilt_speed: float = 0.70,
                 max_zoom_speed: float = 0.50,
                 min_cmd_interval: float = 0.08,
                 # --- autotracker tuning ---
                 yaw_offset_deg: float = 0.0,         # static alignment offsets
                 tilt_offset_deg: float = 0.0,
                 deadband_deg: float = 1.0,           # do nothing if |az| & |el| < deadband
                 pan_kp_per_deg: float = 0.08,        # 0.08 → 10° error = 0.8 speed
                 tilt_kp_per_deg: float = 0.08,
                 min_effective_speed: float = 0.18,   # overcome stiction on small errors
                 zoom_gate_deg: float = 3.0):         # only zoom when nearly centered
        if not host:
            raise PTZError("Missing host")
        self.client = PTZClient(host, username, password, scheme=scheme, port=port,
                                auth_mode=auth_mode, insecure=insecure, timeout=timeout)
        self.invert_pan = bool(invert_pan)
        self.invert_tilt = bool(invert_tilt)
        self.profile_token = str(profile_token) if profile_token else None
        # Control arbitration + deadman-stop
        self._lock = threading.RLock()
        self._stop_timers = []   # type: list[threading.Timer]
        self._locked_by = None   # e.g., 'auto' when auto-tracker holds PTZ
        # caps + rate limiting
        self._max_pan = float(max(0.05, min(1.0, max_pan_speed)))
        self._max_tilt = float(max(0.05, min(1.0, max_tilt_speed)))
        self._max_zoom = float(max(0.05, min(1.0, max_zoom_speed)))
        self._min_cmd_dt = float(max(0.0, min_cmd_interval))
        self._last_cmd_ts = 0.0
        # autotracker state
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.tilt_offset_deg = float(tilt_offset_deg)
        self.deadband_deg = float(max(0.0, deadband_deg))
        self.pan_kp = float(max(0.0, pan_kp_per_deg))
        self.tilt_kp = float(max(0.0, tilt_kp_per_deg))
        self.min_effective_speed = float(max(0.0, min_effective_speed))
        self.zoom_gate_deg = float(max(0.0, zoom_gate_deg))

        # --- autotracker step (relative move) tuning ---
        # Map az/el error [deg] -> relative step [deg]
        self.step_kp_pan_deg  = 0.40   # 10° error -> 4° pan step
        self.step_kp_tilt_deg = 0.40   # 10° error -> 4° tilt step
        self.step_min_deg     = 0.30   # minimum “felt” step to beat backlash
        self.step_max_deg     = 6.00   # clamp to avoid jerks

        # --- manual zoom helpers / distance→zoom policy (manual-only) ---
        # Absolute zoom soft limits (device-specific; kept wide & safe)
        self.zoom_abs_min = 0.0
        self.zoom_abs_max = 100.0
        # Distance → zoom linear mapping for "auto zoom (distance)" when NOT locked by auto
        # At/inside min_dist => max_zoom, at/above max_dist => min_zoom
        self.autoz_min_dist_m = 4.0
        self.autoz_max_dist_m = 40.0
        self.autoz_min_zoom   = 1.0
        self.autoz_max_zoom   = 10.0

    def absolute(self, pan=None, tilt=None, zoom=None, owner: str | None = None):
        return self.absolute_move(pan=pan, tilt=tilt, zoom=zoom, owner=owner)

    def move_absolute(self, pan=None, tilt=None, zoom=None, owner: str | None = None):
        return self.absolute_move(pan=pan, tilt=tilt, zoom=zoom, owner=owner)

    def relative(self, dpan: float = 0.0, dtilt: float = 0.0, dzoom: float = 0.0, owner: str | None = None):
        return self.relative_move(dpan=dpan, dtilt=dtilt, dzoom=dzoom, owner=owner)

    def move_relative(self, dpan: float = 0.0, dtilt: float = 0.0, dzoom: float = 0.0, owner: str | None = None):
        return self.relative_move(dpan=dpan, dtilt=dtilt, dzoom=dzoom, owner=owner)

    def send(self, kind: str, payload: dict):
        """
        Generic shim so callers can do ctl.send("abs", {...}) or ctl.send("rel", {...}).
        """
        kind = (kind or "").lower()
        p = payload or {}
        if kind in ("abs","absolute","move_absolute"):
            return self.absolute_move(pan=p.get("pan"), tilt=p.get("tilt"), zoom=p.get("zoom"))
        if kind in ("rel","relative","move_relative"):
            return self.relative_move(dpan=p.get("dpan",0.0), dtilt=p.get("dtilt",0.0), dzoom=p.get("dzoom",0.0))
        if kind in ("zoom","zoom_burst"):
            return self.zoom_burst(vz=float(p.get("v", p.get("vz", 0.0))), duration=float(p.get("sec", 0.25)))
        if kind in ("stop","halt"):
            return self.stop()
        raise PTZError(f"Unknown send kind: {kind}")

    def is_settled(self, samples: int = 2, sleep_s: float = 0.15) -> bool:
        """
        Heuristic: consider the PTZ 'settled' if pansp and tiltsp are ~0 for N samples.
        Useful after homing, if the caller wants to wait for motion to stop.
        """
        samples = max(1, int(samples))
        ok = 0
        for _ in range(samples):
            st = self.client.status_parsed()
            pansp = float(st.get("pansp", 0.0) or 0.0)
            tiltsp = float(st.get("tiltsp", 0.0) or 0.0)
            if abs(pansp) <= 1.0 and abs(tiltsp) <= 1.0:
                ok += 1
            time.sleep(max(0.05, float(sleep_s)))
        return ok >= samples

    def lock(self, owner: str = "auto"):
        """Prevent manual PTZ commands while locked. Call unlock() to release."""
        with self._lock:
            self._locked_by = owner or "auto"

    def unlock(self, owner: str | None = None):
        """Release lock if owner matches (or force when owner is None)."""
        with self._lock:
           if owner is None or self._locked_by == owner:
                self._locked_by = None

    def is_locked(self) -> bool:
        with self._lock:
            return bool(self._locked_by)
        
    def _ensure_not_locked(self, owner: str | None = None):
        # Allow commands from whoever holds the lock; block others.
        with self._lock:
            if self._locked_by and (owner is None or owner != self._locked_by):
                raise PTZError("PTZ is currently controlled by auto-tracking; disable auto-track to nudge manually.")

    def _norm_to_speed(self, v: float) -> int:
        try:
            v = float(v)
        except Exception:
            v = 0.0
        v = max(-1.0, min(1.0, v))
        return int(round(v * 100))

    def continuous_move(self, vx: float = 0.0, vy: float = 0.0, duration: float = 0.18,
                        zoom: float = 0.0, owner: str | None = None):
        """
        Non-blocking nudge using Illustra ContinuousMove.
        - vx: right +, left - (pan)
        - vy: up +, down -   (tilt)
        - duration: seconds to auto-stop; if <=0, no auto-stop is scheduled.
        - zoom: normalized [-1..1]
        """
        self._ensure_not_locked(owner=owner)
        try:
            duration = float(duration)
        except Exception:
            duration = 0.25
        if duration > 0:
            duration = max(0.05, min(1.5, duration))
        # Rate limit: if we were called too soon, just refresh the deadman timer but
        # avoid hammering firmware with a new request.
        now = time.time()
        if self._min_cmd_dt and (now - self._last_cmd_ts) < self._min_cmd_dt:
            pass  # still refresh deadman below
        else:
            px = self._norm_to_speed((+vx if not self.invert_pan else -vx))
            py = self._norm_to_speed((+vy if not self.invert_tilt else -vy))
            pz = self._norm_to_speed(zoom)
            # Apply caps
            px = int(max(-100, min(100, px * self._max_pan)))
            py = int(max(-100, min(100, py * self._max_tilt)))
            pz = int(max(-100, min(100, pz * self._max_zoom)))
            # Gate zoom while slewing hard in pan/tilt (prevents over-zoom during catch-up)
            if owner == "auto":
                if abs(px) > 25 or abs(py) > 25:
                    pz = 0
            self.client.continuous(pansp=px, tiltsp=py, zoomsp=pz)
            self._last_cmd_ts = now
        # Arm/refresh deadman timer
        self._schedule_stop_burst(duration)

    def _schedule_stop_burst(self, duration: float):
        """Schedule a robust multi-step stop to kill any firmware drift."""
        if not duration or duration <= 0:
            return
        d1 = float(max(0.05, min(1.5, duration)))
        d2 = d1 + 0.18  # second safety stop a bit later
        def _burst():
            # Stop → zero → tiny wait → Stop
            try:
                self.client.stop_hard()
            except Exception:
                pass
        def _burst2():
            # second pass for stubborn drift
            try:
                self.client.stop_hard()
            except Exception:
                pass
        with self._lock:
            # cancel any existing timers
            for t in self._stop_timers:
                with contextlib.suppress(Exception): t.cancel()
            self._stop_timers.clear()
            t1 = threading.Timer(d1, _burst);  t1.daemon = True; t1.start()
            t2 = threading.Timer(d2, _burst2); t2.daemon = True; t2.start()
            self._stop_timers.extend([t1, t2])

    def stop(self, hard: bool = True):
        with self._lock:
            for t in self._stop_timers:
                with contextlib.suppress(Exception): t.cancel()
            self._stop_timers.clear()
        # perform a robust stop burst (twice) to quench drift
        try:
            if hard:
                self.client.stop_hard()
                time.sleep(0.06)
                self.client.stop_hard()
            else:
                self.client.stop()
                time.sleep(0.04)
                # zero speeds as an extra guard
                with contextlib.suppress(Exception):
                    self.client.continuous(pansp=0, tiltsp=0, zoomsp=0)
                self.client.stop()
        except Exception:
            pass

    def goto_preset(self, token):
        self.client.preset_goto(token)

    def relative_move(self, dpan: float = 0.0, dtilt: float = 0.0, dzoom: float = 0.0, owner: str | None = None):
        self._ensure_not_locked(owner=owner)
        return self.client.relative(dpan, dtilt, dzoom)

    def absolute_move(self, pan=None, tilt=None, zoom=None, owner: str | None = None):
        self._ensure_not_locked(owner=owner)
        return self.client.absolute(pan=pan, tilt=tilt, zoom=zoom)

    def zoom_burst(self, vz: float, duration: float = 0.25, owner: str | None = None):
        """
        Convenience for auto-tracker: zoom only, with deadman.
        """
        self.continuous_move(vx=0.0, vy=0.0, duration=duration, zoom=vz, owner=owner)

    def zoom_nudge(self, vz: float, duration: float = 0.25, owner: str | None = None):
        """
        Manual +/− zoom burst (normalized vz in [-1..1]) with deadman stop.
        Blocked when auto-tracker holds the lock (same as other manual moves).
        """
        return self.continuous_move(vx=0.0, vy=0.0, duration=duration, zoom=vz, owner=owner)

    def zoom_to(self, zoom_abs: float, owner: str | None = None):
        """
        Absolute zoom move. Clamped to soft limits. Blocked when auto-tracker holds the lock.
        """
        self._ensure_not_locked(owner=owner)
        try:
            z = float(zoom_abs)
        except Exception:
            z = 0.0
        z = max(self.zoom_abs_min, min(self.zoom_abs_max, z))
        return self.client.absolute(zoom=z)

    def _dist_to_zoom(self, dist_m: float) -> float:
        """
        Linear map: closer => higher zoom. At/inside min_dist -> max_zoom; at/above max_dist -> min_zoom.
        """
        try:
            d = float(dist_m)
        except Exception:
            d = self.autoz_max_dist_m
        dmin, dmax = float(self.autoz_min_dist_m), float(self.autoz_max_dist_m)
        zmin, zmax = float(self.autoz_min_zoom), float(self.autoz_max_zoom)
        if d <= dmin:
            return zmax
        if d >= dmax:
            return zmin
        # interpolate with inverse slope
        t = (dmax - d) / max(0.001, (dmax - dmin))
        return zmin + t * (zmax - zmin)

    def zoom_auto_from_distance(self, dist_m: float, owner: str | None = None):
        """
        Compute target absolute zoom from distance and command an absolute move.
        This is disabled while auto-tracker is locked (same rule as manual).
        """
        self._ensure_not_locked(owner=owner)
        target = self._dist_to_zoom(dist_m)
        return self.zoom_to(target, owner=owner)

    def auto_zoom_burst(self, vz: float, az_deg: float, el_deg: float,
                         duration: float = 0.25, owner: str | None = "auto"):
        """
        Zoom burst that triggers only when aim error is small (centered),
        to avoid over-zooming during pan/tilt catch-up.
          - vz: normalized zoom speed in [-1..1]  (+ = zoom in, − = zoom out)
          - az_deg/el_deg: current azimuth/elevation errors (deg), right/up positive
        Uses self.zoom_gate_deg as the gate. No-ops if locked by another owner.
        """
        # Apply same static offsets that auto_track_step() uses
        az = float(az_deg) + self.yaw_offset_deg
        el = float(el_deg) + self.tilt_offset_deg
        # Gate: only zoom when nearly centered
        if abs(az) > self.zoom_gate_deg or abs(el) > self.zoom_gate_deg:
            return {"skipped": True, "reason": "not_centered", "az": az, "el": el}
        try:
            self._ensure_not_locked(owner=owner)
            self.zoom_burst(vz=max(-1.0, min(1.0, float(vz))), duration=float(duration), owner=owner)
            return {"skipped": False, "az": az, "el": el, "vz": float(vz)}
        except Exception:
            return {"skipped": True, "reason": "error"}

    def vz_from_fov_error(self, fov_current_deg: float, fov_target_deg: float, gain: float = 0.02) -> float:
        """
        Map (current_fov - target_fov) → small zoom speed in [-1..1].
        Positive output = zoom in (narrow FOV). Keep gain tiny to avoid oscillations.
        """
        try:
            err = float(fov_current_deg) - float(fov_target_deg)
            return max(-1.0, min(1.0, gain * err))
        except Exception:
            return 0.0

    # ---------- autotracker helpers ----------
    def set_offsets(self, yaw_offset_deg: float = 0.0, tilt_offset_deg: float = 0.0):
        """Set static boresight offsets (positive = right/up)."""
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.tilt_offset_deg = float(tilt_offset_deg)

    def _clip_norm(self, v: float) -> float:
        return max(-1.0, min(1.0, float(v)))

    def auto_track_step(self, az_deg: float, el_deg: float, dist_m: float | None = None,
                        owner: str | None = "auto", duration: float = 0.20):
        """
        Driftless autotrack: convert az/el error (deg) to small RELATIVE moves (deg).
        az>0 → pan right, el>0 → tilt up. No continuous speeds, so no drift.
        """
        # Apply static offsets
        az = float(az_deg) + self.yaw_offset_deg
        el = float(el_deg) + self.tilt_offset_deg

        # Deadband: if nearly centered, do nothing
        if abs(az) < self.deadband_deg and abs(el) < self.deadband_deg:
            return {"stopped": True, "reason": "deadband", "az": az, "el": el, "mode": "relative"}

        # Proportional step (deg), with min+max clamps to beat stiction/backlash
        def _step(err_deg: float, kp: float) -> float:
            raw = kp * err_deg
            if raw == 0.0:
                return 0.0
            mag = min(self.step_max_deg, max(self.step_min_deg, abs(raw)))
            return math.copysign(mag, raw)

        dpan  = _step(az, self.step_kp_pan_deg)
        dtilt = _step(el, self.step_kp_tilt_deg)

        # Respect inversion flags
        if self.invert_pan:
            dpan = -dpan
        if self.invert_tilt:
            dtilt = -dtilt

        # Zoom policy: keep 0 here to avoid any zoom-induced drift. Main can manage zoom.
        try:
            self._ensure_not_locked(owner=owner)
            self.client.relative(dpan=dpan, dtilt=dtilt, dzoom=0.0)
        except Exception:
            # as a fallback, try a tiny absolute “hold” at current pose (harmless if unsupported)
            with contextlib.suppress(Exception):
                self.client.absolute()  # noop on most builds
        return {"stopped": False, "az": az, "el": el, "dpan": dpan, "dtilt": dtilt, "mode": "relative"}

# ---------- GUI ----------
class PTZJoystickGUI(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.title("PTZ Joystick")
        self.resizable(False, False)
        self.ctl = controller
        self.client = controller.client

        # layout
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # top row: status + refresh
        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="pan=?, tilt=?, zoom=?")
        ttk.Label(top, textvariable=self.status_var, width=36).pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_status).pack(side="left", padx=(8,0))

        # middle: joystick + zoom column + arrows
        mid = ttk.Frame(self, padding=(8,0))
        mid.grid(row=1, column=0)

        # joystick canvas
        self.size = 220
        self.radius = 85
        self.center = self.size//2
        self.canvas = tk.Canvas(mid, width=self.size, height=self.size, highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=4, padx=(0,10))
        self._draw_base()
        self.dragging = False
        self.handle = self.canvas.create_oval(self.center-14, self.center-14,
                                              self.center+14, self.center+14, fill="#4ea3ff", outline="")
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.bind_all("<ButtonRelease-1>", lambda e: self.stop_motion())

        # zoom column
        zoomf = ttk.Frame(mid)
        zoomf.grid(row=0, column=1, sticky="ns")
        ttk.Label(zoomf, text="Zoom").pack(pady=(0,2))
        self.zoom_hold = {"dir": 0, "held": False}
        self.btn_zin  = ttk.Button(zoomf, text="+ ZOOM")
        self.btn_zout = ttk.Button(zoomf, text="− ZOOM")
        self.btn_zin.pack(fill="x", pady=2)
        self.btn_zout.pack(fill="x", pady=2)
        self.btn_zin.bind("<ButtonPress-1>",  lambda e: self.start_zoom(+50))
        self.btn_zin.bind("<ButtonRelease-1>",lambda e: self.stop_motion())
        self.btn_zout.bind("<ButtonPress-1>", lambda e: self.start_zoom(-50))
        self.btn_zout.bind("<ButtonRelease-1>",lambda e: self.stop_motion())

        # arrows (nudges)
        arrows = ttk.Frame(mid)
        arrows.grid(row=0, column=2, padx=(10,0), sticky="n")
        ttk.Label(arrows, text="Nudges").grid(row=0, column=0, columnspan=3)
        btn = lambda t, r, c, cmd: ttk.Button(arrows, text=t, width=6, command=cmd).grid(row=r, column=c, padx=2, pady=2)
        btn("↑", 1, 1, lambda: self.nudge(0, +40))
        btn("←", 2, 0, lambda: self.nudge(-40, 0))
        btn("STOP", 2, 1, self.stop_motion)
        btn("→", 2, 2, lambda: self.nudge(+40, 0))
        btn("↓", 3, 1, lambda: self.nudge(0, -40))

        # sensitivity slider
        bot = ttk.Frame(self, padding=8)
        bot.grid(row=2, column=0, sticky="ew")
        ttk.Label(bot, text="Sensitivity").pack(side="left")
        self.sens = tk.DoubleVar(value=1.0)
        ttk.Scale(bot, from_=0.2, to=2.0, variable=self.sens, orient="horizontal", length=180).pack(side="left", padx=8)
        ttk.Button(bot, text="STOP", command=self.stop_motion).pack(side="left", padx=(8,0))

        # presets
        pres = ttk.Frame(self, padding=(8,0,8,8))
        pres.grid(row=3, column=0, sticky="ew")
        self.preset_var = tk.StringVar(value="")
        self.preset_box = ttk.Combobox(pres, textvariable=self.preset_var, width=24, state="readonly")
        self.preset_box.pack(side="left")
        ttk.Button(pres, text="Refresh", command=self.load_presets).pack(side="left", padx=4)
        ttk.Button(pres, text="Goto", command=self.goto_selected).pack(side="left", padx=4)
        self.new_name = tk.StringVar()
        ttk.Entry(pres, textvariable=self.new_name, width=16).pack(side="left", padx=(8,4))
        ttk.Button(pres, text="Save as", command=self.save_preset).pack(side="left")

        # keyboard bindings
        self.bind("<Up>",    lambda e: self.nudge(0, +40))
        self.bind("<Down>",  lambda e: self.nudge(0, -40))
        self.bind("<Left>",  lambda e: self.nudge(-40, 0))
        self.bind("<Right>", lambda e: self.nudge(+40, 0))
        self.bind("+",       lambda e: self.start_zoom(+50))
        self.bind("-",       lambda e: self.start_zoom(-50))
        self.bind("<KeyRelease-plus>", lambda e: self.stop_motion())
        self.bind("<KeyRelease-minus>",lambda e: self.stop_motion())

        # safety stop on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # initial fetch
        self.after(200, self.refresh_status)
        self.after(300, self.load_presets)

    # ---- joystick math ----
    def _draw_base(self):
        r = self.radius
        c = self.center
        self.canvas.create_oval(c-r, c-r, c+r, c+r, outline="#9aa8b0", width=2)
        # crosshairs
        self.canvas.create_line(c-r, c, c+r, c, fill="#d0d7dc")
        self.canvas.create_line(c, c-r, c, c+r, fill="#d0d7dc")

    def on_down(self, e):
        self.dragging = True
        self._move_handle(e.x, e.y)
        self._send_from_handle()

    def on_drag(self, e):
        if not self.dragging: return
        self._move_handle(e.x, e.y)
        self._send_from_handle()

    def on_up(self, e):
        self.dragging = False
        self._center_handle()
        self.stop_motion()

    def _move_handle(self, x, y):
        # clamp inside circle
        dx = x - self.center
        dy = y - self.center
        dist2 = dx*dx + dy*dy
        r = self.radius
        if dist2 > r*r:
            import math
            ang = math.atan2(dy, dx)
            x = self.center + r * math.cos(ang)
            y = self.center + r * math.sin(ang)
        self.canvas.coords(self.handle, x-14, y-14, x+14, y+14)

    def _center_handle(self):
        c = self.center
        self.canvas.coords(self.handle, c-14, c-14, c+14, c+14)

    def _handle_vector(self):
        # normalized -1..1
        x1, y1, x2, y2 = self.canvas.coords(self.handle)
        hx = (x1 + x2) / 2 - self.center
        hy = (y1 + y2) / 2 - self.center
        nx = max(-1.0, min(1.0, hx / self.radius))
        ny = max(-1.0, min(1.0, hy / self.radius))
        return nx, ny

    def _send_from_handle(self):
        nx, ny = self._handle_vector()
        sens = float(self.sens.get())
        # convert to normalized speeds [-1..1] with tilt up positive
        vx = max(-1.0, min(1.0, nx * sens))
        vy = max(-1.0, min(1.0, (-ny) * sens))
        # send with a short deadman so if release is missed, it still stops
        threading.Thread(
            target=self._safe_call,
            args=("move_with_deadman", int(vx * 100), int(vy * 100), 0, 0.35),
            daemon=True
        ).start()
    # ---- actions ----
    def stop_motion(self):
        self.zoom_hold["held"] = False
        threading.Thread(target=self._safe_call, args=("stop",), daemon=True).start()

    def start_zoom(self, zoomsp):
        # repeat while held; each burst has a short deadman
        self.zoom_hold.update({"dir": int(max(-100, min(100, zoomsp))), "held": True})
        def _loop():
            while self.zoom_hold["held"]:
                self._safe_call("move_with_deadman", 0, 0, self.zoom_hold["dir"], 0.30)
                time.sleep(0.20)
        threading.Thread(target=_loop, daemon=True).start()

    def nudge(self, pansp, tiltsp, dur=0.4):
        threading.Thread(
            target=self._safe_call, args=("move_with_deadman", pansp, tiltsp, 0, float(dur)), daemon=True
        ).start()

    def refresh_status(self):
        def _run():
            try:
                t = self.client.status()
                # parse pan/tilt/zoom lines
                vals = {}
                for line in t.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        vals[k.strip()] = v.strip()
                s = f"pan={vals.get('pan','?')}  tilt={vals.get('tilt','?')}  zoom={vals.get('zoom','?')}"
                self.status_var.set(s)
            except Exception as e:
                self.status_var.set("status: error")
        threading.Thread(target=_run, daemon=True).start()

    def load_presets(self):
        def _run():
            try:
                txt = self.client.presets_read()
                names = []
                for ln in txt.splitlines():
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) == 2 and parts[1].startswith("name="):
                        names.append(parts[1][5:])
                self.preset_box["values"] = names
                if names and not self.preset_var.get():
                    self.preset_var.set(names[0])
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def goto_selected(self):
        name = self.preset_var.get().strip()
        if not name: return
        threading.Thread(target=self._safe_call, args=("goto", name), daemon=True).start()

    def save_preset(self):
        nm = self.new_name.get().strip()
        if not nm:
            messagebox.showinfo("Preset", "Enter a name")
            return
        def _run():
            try:
                self.client.preset_create(nm)
                time.sleep(0.2)
                self.load_presets()
                self.preset_var.set(nm)
            except Exception as e:
                messagebox.showerror("Preset", f"Error: {e}")
        threading.Thread(target=_run, daemon=True).start()

    def _safe_call(self, what, *args):
        try:
            if what == "move":
                # legacy: continuous without deadman (not used now)
                pansp, tiltsp, zoomsp = args
                self.ctl.continuous_move(vx=pansp/100.0, vy=tiltsp/100.0, duration=0.0, zoom=zoomsp/100.0)
            elif what == "move_with_deadman":
                pansp, tiltsp, zoomsp, dur = args
                self.ctl.continuous_move(vx=pansp/100.0, vy=tiltsp/100.0, duration=float(dur), zoom=zoomsp/100.0)
            elif what == "stop":
                self.ctl.stop(hard=True)
            elif what == "goto":
                self.client.preset_goto(args[0])
        except requests.HTTPError as he:
            # show last code in title briefly
            self.title(f"PTZ Joystick  (HTTP {getattr(he.response,'status_code','?')})")
        except Exception as e:
            self.title("PTZ Joystick  (err)")

    def _on_close(self):
        try:
            self.client.stop()
        except Exception:
            pass
        self.destroy()

def main():
    ap = argparse.ArgumentParser(description="PTZ control (Illustra iAPI) — GUI + CLI")
    ap.add_argument("--host", required=True, help="Camera IP (e.g. 192.168.40.51)")
    ap.add_argument("--user", required=True)
    ap.add_argument("--pwd", required=True)
    ap.add_argument("--scheme", choices=["http","https"], default="http")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--auth", choices=["auto","basic","digest"], default="auto")
    ap.add_argument("--insecure", action="store_true", help="Ignore TLS cert errors (https)")
    ap.add_argument("--invert-pan", action="store_true", help="Invert pan direction")
    ap.add_argument("--invert-tilt", action="store_true", help="Invert tilt direction")

    sub = ap.add_subparsers(dest="cmd", required=True)
    # GUI
    sub.add_parser("gui", help="Launch joystick GUI")
    # status
    sub.add_parser("status", help="Print PTZ status (pan/tilt/zoom)")
    sub.add_parser("stop", help="Issue hard Stop")
    sub.add_parser("explain", help="Print capability info (Explain)")
    # continuous move (nudge)
    p_cont = sub.add_parser("cont", help="Continuous move/nudge")
    p_cont.add_argument("--vx", type=float, default=0.0, help="Normalized pan speed (-1..1)  right+=+")
    p_cont.add_argument("--vy", type=float, default=0.0, help="Normalized tilt speed (-1..1) up+=+")
    p_cont.add_argument("--sec", type=float, default=0.3, help="Duration seconds before auto-stop")
    p_cont.add_argument("--zoom", type=float, default=0.0, help="Normalized zoom speed (-1..1)")
    p_abs = sub.add_parser("abs", help="Absolute move (degrees / zoom units)")
    p_abs.add_argument("--pan", type=float)
    p_abs.add_argument("--tilt", type=float)
    p_abs.add_argument("--zoom", type=float)
    p_rel = sub.add_parser("rel", help="Relative move (step in degrees / zoom units)")
    p_rel.add_argument("--dpan", type=float, default=0.0)
    p_rel.add_argument("--dtilt", type=float, default=0.0)
    p_rel.add_argument("--dzoom", type=float, default=0.0)
    p_zn = sub.add_parser("zoom", help="Zoom nudge/burst (normalized speed)")
    p_zn.add_argument("--vz", type=float, required=True, help="Normalized zoom speed (-1..1)")
    p_zn.add_argument("--sec", type=float, default=0.25, help="Duration seconds before auto-stop")
    p_za = sub.add_parser("zoom_abs", help="Absolute zoom to a target value")
    p_za.add_argument("--zoom", type=float, required=True, help="Absolute zoom value")
    p_zd = sub.add_parser("zoom_auto", help="Auto zoom (manual-only) from distance")
    p_zd.add_argument("--dist", type=float, required=True, help="Distance in meters")
    # presets
    p_preset = sub.add_parser("preset", help="Preset ops")
    g = p_preset.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="List presets")
    g.add_argument("--goto", metavar="TOKEN", help="Go to preset (id or name)")
    g.add_argument("--save", metavar="NAME", help="Create preset with NAME")
    g.add_argument("--delete", metavar="TOKEN", help="Delete preset (id or name)")
    # autotracker single step (for testing from shell)
    p_track = sub.add_parser("track", help="One autotrack step (map az/el to move)")
    p_track.add_argument("--az", type=float, required=True, help="Azimuth error in degrees (right+=+)")
    p_track.add_argument("--el", type=float, required=True, help="Elevation error in degrees (up+=+)")
    p_track.add_argument("--dist", type=float, default=None, help="Range in meters (optional)")
    p_track.add_argument("--sec", type=float, default=0.20, help="Burst duration seconds")

    args = ap.parse_args()

    # Build both client + controller (controller used by CLI 'cont')
    client = PTZClient(args.host, args.user, args.pwd,
                       scheme=args.scheme, port=args.port,
                       auth_mode=args.auth, insecure=args.insecure)
    ctl = PTZController(args.host, args.user, args.pwd,
                        scheme=args.scheme, port=args.port,
                        auth_mode=args.auth, insecure=args.insecure,
                        invert_pan=args.invert_pan, invert_tilt=args.invert_tilt)

    if args.cmd == "gui":
        app = PTZJoystickGUI(ctl)
        app.mainloop()
        return

    if args.cmd == "status":
        try:
            print(client.status().strip())
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "explain":
        try:
            print(client.explain().strip())
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "stop":
        try:
            ctl.stop()
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "cont":
        try:
            ctl.continuous_move(vx=args.vx, vy=args.vy, duration=args.sec, zoom=args.zoom)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "abs":
        try:
            ctl.absolute_move(pan=args.pan, tilt=args.tilt, zoom=args.zoom)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "rel":
        try:
            ctl.relative_move(dpan=args.dpan, dtilt=args.dtilt, dzoom=args.dzoom)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "zoom":
        try:
            ctl.zoom_nudge(vz=args.vz, duration=args.sec)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return
    if args.cmd == "zoom_abs":
        try:
            ctl.zoom_to(zoom_abs=args.zoom)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return
    if args.cmd == "zoom_auto":
        try:
            ctl.zoom_auto_from_distance(dist_m=args.dist)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "preset":
        try:
            if args.list:
                print(client.presets_read().strip())
            elif args.goto:
                client.preset_goto(args.goto)
                print("OK")
            elif args.save:
                client.preset_create(args.save)
                print("OK")
            elif args.delete:
                client.preset_delete(args.delete)
                print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.cmd == "track":
        try:
            ctl.auto_track_step(az_deg=args.az, el_deg=args.el, dist_m=args.dist, owner="auto", duration=float(args.sec))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

if __name__ == "__main__":
    main()

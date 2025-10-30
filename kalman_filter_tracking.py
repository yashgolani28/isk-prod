import time
import uuid
import numpy as np
from collections import deque
from datetime import datetime

_RADIAL_FLOOR_MS = 0.20
_LAMBDA_M = 0.00494  # 60 GHz wavelength approx 5 mm
_MAX_RANGE_M     = 150.0     # drop detections beyond this LOS range
_MIN_RANGE_M     = 0.20      # drop "on top of sensor" ghosts
_MAX_RADIAL_MPS  = 80.0      # ~288 km/h, clamp fused radial
_MAX_SPEED_MPS   = 80.0      # clamp final 3D speed magnitude
_LOW_SNR_MARGIN  = 3.0       # allow a small margin below min_snr_db

class KalmanFilter3D:
    """Simple constant-velocity 3D Kalman filter."""
    def __init__(self, initial_position, initial_velocity=None):
        self.state = np.zeros((6, 1), dtype=float)  # [x,y,z,vx,vy,vz]
        self.state[:3, 0] = np.array(initial_position, dtype=float).reshape(3)
        if initial_velocity is not None:
            self.state[3:, 0] = np.array(initial_velocity, dtype=float).reshape(3)

        self.P = np.eye(6, dtype=float) * 1.0
        self.Q = np.eye(6, dtype=float) * 0.05
        self.R = np.eye(3, dtype=float) * 0.15
        self.F = np.eye(6, dtype=float)
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        self.last_update = time.time()
        self.dt = 0.066  # ~15 Hz default

    def _build_F(self, dt):
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def predict(self, dt=None):
        if dt is None:
            dt = max(0.0, time.time() - self.last_update)
        else:
            dt = max(0.0, float(dt))
        if dt == 0:
            return

        self.F = self._build_F(dt)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.last_update = time.time()

    def update(self, z):
        """Measurement update with position z=(x,y,z)."""
        z = np.array(z, dtype=float).reshape(3, 1)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        self.last_update = time.time()

    def get_state(self):
        pos = self.state[:3, 0]
        vel = self.state[3:, 0]
        return pos.copy(), vel.copy()


class ObjectTracker:
    """
    Stable ID association + 3D Kalman smoothing.
    - Associates by (1) stable radar TID map, else (2) nearest neighbor of predicted positions.
    - Never overwrites matched tracker ID with incoming detection id.
    """
    def __init__(self, speed_limit_kmh=1, speed_limits_map=None,
                 unambiguous_mps=None,   # v_unamb for Doppler wrap, m/s
                 fusion_alpha=0.6,       # Doppler vs range-rate weighting
                 max_vehicle_accel=4.0,  # m/s^2, plausibility for unwrap
                 moving_threshold_kmh=0.8,
                 stationary_threshold_kmh=0.3,
                 min_hits=2,
                 write_synthetic_doppler=False,
                 radial_floor_ms=_RADIAL_FLOOR_MS,
                 min_snr_db=5.0,
                 min_signal_level=1.0,
                 unwrap_k_max=2,
                 road_axis_unit=None,    # unit vector of road direction in radar coords
                 cos_guard=0.55,        # |cos(theta)| below this → fall back logic
                 max_range_m=150.0,
                 min_range_m=0.20,
                 max_radial_mps=80.0,
                 max_speed_mps=80.0,
                 low_snr_margin_db=3.0,
                 strict_radial_only=False,
                 disable_unwrap_for_classes=None):
        self.trackers = {}     # TRK_xxx -> KalmanFilter3D
        self.history = {}
        self.speed_limit_kmh = speed_limit_kmh
        self.speed_limits_map = speed_limits_map or {"default": speed_limit_kmh}
        self.max_age = 4.0
        self.match_tolerance = 0.6
        self.logged_ids = set()

        self.tid_map = {}      # "TID_12" -> "TRK_00001"
        self.tid_seen = {}     # last time that TID was seen
        self.last_time = {}    # last update timestamp per TRK
        self.hits = {}         # consecutive detections per TRK
        self.confirmed = {}    # TRK -> bool (true once min_hits reached or stable TID seen)
        self.last_range = {}   # last LOS range per TRK (meters)
        self.last_radial = {}  # last radial speed per TRK (m/s)
        self.seq = 0

        self.innovation_gate_pos = 1.0
        self.innovation_gate_vel = 2.0

        # unwrap/fusion params
        self.v_unamb_ms = float(unambiguous_mps) if unambiguous_mps else None
        self.fusion_alpha = float(fusion_alpha)
        self.a_max = float(max_vehicle_accel)
        self.min_snr_db = float(min_snr_db)
        self.min_signal_level = float(min_signal_level)
        self.unwrap_k_max = int(unwrap_k_max)

        # ---- Road-speed settings ----
        self.moving_threshold_kmh = float(moving_threshold_kmh)
        self.stationary_threshold_kmh = float(stationary_threshold_kmh)
        self.min_hits = int(min_hits)
        self.write_synthetic_doppler = bool(write_synthetic_doppler)
        self.radial_floor_ms = float(radial_floor_ms)
        self.road_axis_unit = self._normalize_axis(
            np.asarray(road_axis_unit, dtype=float)
        ) if road_axis_unit is not None else np.array([0.0, 1.0, 0.0], dtype=float)  # default: +Y is “down-road”
        self.cos_guard = float(cos_guard)

        # ---- Plausibility caps ----
        self.max_range_m = float(max_range_m)
        self.min_range_m = float(min_range_m)
        self.max_radial_mps = float(max_radial_mps)
        self.max_speed_mps = float(max_speed_mps)
        self.low_snr_margin_db = float(low_snr_margin_db)
        self.strict_radial_only = bool(strict_radial_only)
        self.disable_unwrap_for_classes = set(x.upper() for x in (disable_unwrap_for_classes or set()))

    @staticmethod
    def _finite(x, default=0.0):
        try:
            xf = float(x)
            return xf if np.isfinite(xf) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _normalize_axis(v):
        try:
            n = float(np.linalg.norm(v))
            if n > 0:
                return (v / n).astype(float)
        except Exception:
            pass
        # default forward-Y if anything odd happens
        return np.array([0.0, 1.0, 0.0], dtype=float)

    def set_road_axis(self, vec=None, yaw_deg=None):
        """
        Set the road direction as a unit vector in radar coordinates.
        If yaw_deg is given, rotate +Y by yaw about +Z (right-handed).
        """
        if yaw_deg is not None:
            r = np.radians(float(yaw_deg))
            # rotate [0,1,0] by yaw around Z → [sin(yaw), cos(yaw), 0]
            vec = np.array([np.sin(r), np.cos(r), 0.0], dtype=float)
        if vec is not None:
            self.road_axis_unit = self._normalize_axis(np.asarray(vec, dtype=float))
        return self.road_axis_unit.copy()

    def set_unambiguous_speed(self, v_ms):
        """Call this after applying radar cfg; keeps tracker decoupled from RF."""
        try:
            self.v_unamb_ms = float(v_ms)
        except Exception:
            self.v_unamb_ms = None

    def _unwrap_velocity_radial(self, v_meas, v_prev, v_unamb, a_max, dt, *, prefer_k0=False, kmax_override=None):
        """Choose wrapped candidate closest/physically-plausible to previous."""
        if v_unamb is None or v_unamb <= 0 or dt <= 0:
            return float(v_meas), 0  
        kmax = max(1, int(kmax_override if kmax_override is not None else self.unwrap_k_max))
        cands = [(v_meas + k*v_unamb, k) for k in range(-kmax, kmax + 1)]
        def score(v):
            a = abs((v - v_prev) / dt)
            return abs(v - v_prev) + max(0.0, a - a_max) * 10.0
        v_best, k_best = min(cands, key=lambda vk: score(vk[0]))
        if prefer_k0 and k_best != 0:
            if abs(v_meas - v_prev) <= 0.6 * v_unamb:
                v_best, k_best = float(v_meas), 0
        return float(v_best), int(k_best)

    @staticmethod
    def _alpha_from_quality(alpha_base, snr, sig, min_snr_db):
        """SNR-aware fusion weight: ~0 below threshold, ~alpha_base when SNR is healthy."""
        if not np.isfinite(snr) or snr <= 0:
            return 0.0
        # logistic ramp centered near (min_snr_db + ~1.5 dB)
        x = (snr - (min_snr_db + 1.5)) / 3.0
        w = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(alpha_base * w, 0.0, 1.0))

    def get_limit_for(self, obj_type):
        return self.speed_limits_map.get(str(obj_type).upper(),
                                         self.speed_limits_map.get("default", self.speed_limit_kmh))

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _extract_position(det):
        if all(k in det for k in ("x", "y", "z")):
            return float(det["x"]), float(det["y"]), float(det["z"])
        # fallback: spherical-ish reconstruction
        r  = float(det.get("distance", 0.0))
        az = np.radians(float(det.get("azimuth", 0.0)))
        el = np.radians(float(det.get("elevation", 0.0)))
        # x = r * sin(az) * cos(el); y = r * cos(az) * cos(el); z = r * sin(el)
        x = r * np.sin(az) * np.cos(el)
        y = r * np.cos(az) * np.cos(el)
        z = r * np.sin(el)
        return float(x), float(y), float(z)

    def calculate_score(self, det):
        # Prefer speed along the road if available (better for violation logic)
        speed = float(det.get("speed_along_road_kmh", det.get("speed_kmh", 0.0)))
        sig = float(det.get("signal_level", 0.0))
        conf = float(det.get("confidence", 0.5))
        limit = max(self.get_limit_for(det.get("type", "default")), 1e-3)
        speed_ratio = min(speed / limit, 3.0)
        return float(0.6 * speed_ratio + 0.3 * conf + 0.1 * np.tanh(sig / 10.0))

    @staticmethod
    def _predict_peek(kf, dt):
        """Peek next state without mutating filter."""
        A = np.eye(6)
        for i in range(3):
            A[i, i+3] = max(dt, 0.0)
        state = A @ kf.state
        return state[:3, 0], state[3:, 0]

    def update_tracks(self, detections, yolo_detections=None, frame_timestamp=None):
        current_time = frame_timestamp if frame_timestamp is not None else time.time()
        updated_objects = []

        # Retire stale trackers
        alive = {}
        for oid, kf in self.trackers.items():
            if current_time - kf.last_update < self.max_age:
                alive[oid] = kf
            else:
                # prune bookkeeping for dead tracks
                self.last_time.pop(oid, None)
                self.last_range.pop(oid, None)
                self.last_radial.pop(oid, None)
                self.hits.pop(oid, None)
                self.confirmed.pop(oid, None)
        self.trackers = alive
        # Also prune stale radar TIDs so the map doesn’t grow unbounded
        cutoff = current_time - self.max_age
        for tid, ts in list(self.tid_seen.items()):
            if ts < cutoff:
                self.tid_seen.pop(tid, None)
                self.tid_map.pop(tid, None)
        for det in detections:
            # position required
            if not any(k in det for k in ("x","y","z","distance")):
                continue

            position = self._extract_position(det)
            rng_meas = float(np.linalg.norm(np.array(position, dtype=float)))
            if (not np.isfinite(rng_meas)) or (rng_meas < self.min_range_m) or (rng_meas > self.max_range_m):
                continue

            # Ultra-low SNR + low signal? Skip (noise burst)
            snr_db = self._finite(det.get("snr", 0.0))
            sig_lv = self._finite(det.get("signal_level", det.get("snr_lin", 0.0)))
            if (snr_db < (self.min_snr_db - self.low_snr_margin_db)) and (sig_lv < self.min_signal_level):
                continue

            # Stable radar TID if available
            source = str(det.get("source", "")).lower()
            raw_id = det.get("source_id") or det.get("id")
            if source == "pointcloud":
                raw_id = None

            if isinstance(raw_id, (int, float)):
                stable_key = f"TID_{int(raw_id)}"
            elif isinstance(raw_id, str) and raw_id.startswith("TID_"):
                stable_key = raw_id
            else:
                stable_key = None

            matched_id = None
            if stable_key and stable_key in self.tid_map and self.tid_map[stable_key] in self.trackers:
                matched_id = self.tid_map[stable_key]
            else:
                # nearest neighbor on predicted position
                best = (None, float("inf"))
                for oid, kf in self.trackers.items():
                    last_t = self.last_time.get(oid, current_time)
                    dt = max(0.0, float(current_time - last_t))
                    pos_pred, _ = self._predict_peek(kf, dt)
                    dist = self._distance(pos_pred, position)
                    if dist < self.match_tolerance and dist < best[1]:
                        best = (oid, dist)
                matched_id = best[0]

            meas_v = [
                det.get("vx", det.get("velX", 0.0)),
                det.get("vy", det.get("velY", 0.0)),
                det.get("vz", det.get("velZ", 0.0)),
            ]
            try:
                meas_v = [float(meas_v[0]), float(meas_v[1]), float(meas_v[2])]
            except Exception:
                meas_v = [0.0, 0.0, 0.0]

            # Create tracker when needed
            if not matched_id:
                self.seq += 1
                matched_id = f"TRK_{self.seq:05d}"
                self.trackers[matched_id] = KalmanFilter3D(position, initial_velocity=meas_v)
                self.last_time[matched_id] = float(current_time)
                self.hits[matched_id] = 0
                self.confirmed[matched_id] = bool(stable_key)

            # Maintain TID→TRK map
            if stable_key:
                self.tid_map[stable_key] = matched_id
                self.tid_seen[stable_key] = float(current_time)
                self.confirmed[matched_id] = True

            # Predict/update
            kf = self.trackers[matched_id]
            last_t = self.last_time.get(matched_id, current_time)
            dt = max(0.0, float(current_time - last_t))
            pos_pred, vel_pred = self._predict_peek(kf, dt)

            innov_pos = float(np.linalg.norm(np.array(position) - np.array(pos_pred)))
            innov_vel = float(np.linalg.norm(np.array(meas_v) - np.array(vel_pred)))
            if innov_pos < self.innovation_gate_pos or innov_vel < self.innovation_gate_vel:
                kf.predict(dt)
                kf.update(position)
                self.last_time[matched_id] = float(current_time)
                self.hits[matched_id] = self.hits.get(matched_id, 0) + 1
                if not self.confirmed.get(matched_id, False) and self.hits[matched_id] >= self.min_hits:
                    self.confirmed[matched_id] = True
            else:
                # still advance time in filter so age is right
                kf.predict(dt)
                self.last_time[matched_id] = float(current_time)

            # --- Velocity fusion with Doppler unwrap + range-rate ---
            pos, vel = kf.get_state()      # current KF state after update
            pos = np.array(pos, dtype=float)
            vel = np.array(vel, dtype=float)
            meas_v_vec = np.array(meas_v, dtype=float)

            # LOS & ranges
            rng_now = float(np.linalg.norm(pos))
            los = (pos / rng_now) if rng_now > 1e-6 else np.array([0.0, 1.0, 0.0])  # default forward-Y
            rng_prev = self.last_range.get(matched_id, rng_now)
            v_rr = -(rng_now - rng_prev) / dt if dt > 1e-6 else 0.0  # m/s (radial)

            # Raw Doppler radial from measurement:
            v_dopp_meas = float(det.get("radial_velocity", np.dot(meas_v_vec, los)))
            v_rad_prev  = self.last_radial.get(matched_id, float(np.dot(vel_pred, los)))

            obj_type = str(det.get('type', 'DEFAULT')).upper()
            limit_kmh = float(self.get_limit_for(obj_type))
            limit_mps = max(0.5, limit_kmh / 3.6)
            low_speed_context = (limit_kmh <= 15.0)  # humans, walkers, bicycles
            kmax_override = 1 if low_speed_context else None
            prefer_k0 = low_speed_context or (abs(v_dopp_meas) < 1.0)  # ~<1 m/s raw → prefer no unwrap
            use_strict = bool(self.strict_radial_only) or (obj_type in self.disable_unwrap_for_classes)
 
            v_unwrapped, unwrap_k = self._unwrap_velocity_radial(
                v_dopp_meas, v_rad_prev, self.v_unamb_ms, self.a_max, dt if dt>0 else 1e-3,
                prefer_k0=prefer_k0, kmax_override=kmax_override
            )
            if use_strict:
                v_unwrapped, unwrap_k = float(v_rr), 0

            # Fuse unwrapped Doppler with range-rate
            snr = float(det.get("snr", 0.0))
            sig = float(det.get("signal_level", 0.0))
            alpha = self._alpha_from_quality(self.fusion_alpha, snr, sig, self.min_snr_db)
            v_rad_fused = alpha * v_unwrapped + (1.0 - alpha) * v_rr
            if use_strict:
                v_rad_fused = float(v_rr)
            v_rad_cap = 3.0 * limit_mps
            if np.isfinite(v_rad_fused):
                v_rad_fused = float(np.clip(v_rad_fused, -v_rad_cap, v_rad_cap))
            if not np.isfinite(v_rad_fused):
                v_rad_fused = 0.0
            v_rad_fused = float(np.clip(v_rad_fused, -self.max_radial_mps, self.max_radial_mps))

            # Project KF velocity onto LOS to enforce fused radial, keep transverse
            if not use_strict:
                v_proj = float(np.dot(vel, los))
                vel_adj = vel + (v_rad_fused - v_proj) * los
                kf.state[3:, 0] = np.asarray(vel_adj, dtype=float).reshape(3)
            else:
                vel_adj = vel
            vm = float(np.linalg.norm(kf.state[3:, 0]))
            if (not np.isfinite(vm)) or (vm > self.max_speed_mps):
                if vm > 1e-9:
                    kf.state[3:, 0] *= (self.max_speed_mps / vm)
                else:
                    kf.state[3:, 0] = np.array([0.0, 0.0, 0.0], dtype=float)
            vel = kf.state[3:, 0].copy()

            # book-keeping for next frame
            self.last_range[matched_id] = rng_now
            self.last_radial[matched_id] = v_rad_fused
            det['unwrap_k'] = int(unwrap_k)
            det['v_rad_rr'] = float(v_rr)
            det['v_rad_unwrapped'] = float(v_unwrapped)
            det['v_unamb'] = float(self.v_unamb_ms) if self.v_unamb_ms else None
            det['alpha_used'] = float(alpha)

            vel_mag = float(np.linalg.norm(vel))
            vel_cap = 3.0 * limit_mps
            if np.isfinite(vel_mag) and vel_mag > vel_cap:
                scale = vel_cap / (vel_mag + 1e-9)
                vel = vel * scale
                vel_mag = float(np.linalg.norm(vel))
            speed_kmh = float(vel_mag * 3.6)

            # -------- Road-speed correction (cosine-angle) ----------
            # cos(theta) = dot(road_axis, LOS). If |cos| small, division explodes;
            # guard by falling back to projecting full 3D velocity onto road axis.
            road_axis = self.road_axis_unit
            cos_lr = float(np.dot(road_axis, los))
            if use_strict:
                # strict mode: never divide by cos; be conservative (radial as road)
                v_road = v_rad_fused
                v_road_method = "radial/strict"
            else:
                use_vector_proj = (abs(v_rad_fused) < self.radial_floor_ms) or (abs(cos_lr) < self.cos_guard)
                if not use_vector_proj:
                    v_road = v_rad_fused / cos_lr
                    v_road_method = "radial/cos"
                else:
                    v_road = float(np.dot(vel, road_axis))
                    v_road_method = "vector-proj"
            if not np.isfinite(v_road):
                v_road = 0.0
            v_road = float(np.clip(v_road, -self.max_speed_mps, self.max_speed_mps))
            speed_along_road_kmh = float(abs(v_road) * 3.6)
            angle_los_road_deg = float(np.degrees(np.arccos(np.clip(cos_lr, -1.0, 1.0))))

            # Class sanity cap to suppress nonsense in UI/score
            _typ = str(det.get("type", "")).upper()
            if _typ == "HUMAN" and speed_along_road_kmh > 35.0:
                speed_along_road_kmh = 0.0
                v_road = 0.0

            rng = float(np.clip(self._finite(rng_now), self.min_range_m, self.max_range_m))
            v_radial = float(np.dot(vel, los)) if rng > 1e-6 else 0.0
            df_meas = det.get("doppler_frequency", det.get("doppler", None))
            if df_meas is None and self.write_synthetic_doppler:
                doppler = float((2.0 * v_radial) / _LAMBDA_M)
            else:
                doppler = float(df_meas) if df_meas is not None else 0.0

            snr = float(det.get("snr", 0.0))
            gain = float(det.get("gain", 1.0))
            snr_lin = 10.0**(float(det.get("snr", 0.0))/10.0)
            signal_level = float(gain * snr_lin) if (gain and snr_lin) else 0.0

            spd_for_motion = speed_along_road_kmh if np.isfinite(speed_along_road_kmh) else speed_kmh
            motion_state = str(det.get('motion_state', 'unknown')).upper()
            if motion_state in ('UNKNOWN', 'STATIC', 'STATIONARY'):
                if spd_for_motion > self.moving_threshold_kmh:
                    motion_state = 'MOVING'
                elif spd_for_motion < self.stationary_threshold_kmh:
                    motion_state = 'STATIONARY'
                det['motion_state'] = motion_state

            # Direction fallback
            direction = det.get("direction")
            if not direction or direction == "unknown":
                v_along_road = float(np.dot(vel, self.road_axis_unit))
                if v_along_road < -0.05:
                    direction = "TOWARDS"
                elif v_along_road > 0.05:
                    direction = "AWAY"
                else:
                    direction = "STATIC"

            # Stable ID back onto detection (do NOT overwrite with incoming)
            if not self.confirmed.get(matched_id, False):
                # keep filter running internally, but do not surface to callers
                continue
            det['object_id'] = matched_id
            if stable_key:
                det['source_id'] = stable_key

            det.update({
                'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                'vx': float(vel[0]), 'vy': float(vel[1]), 'vz': float(vel[2]),
                'velX': float(vel[0]), 'velY': float(vel[1]), 'velZ': float(vel[2]),
                'velocity': float(vel_mag),
                'speed_radial_kmh': float(abs(v_rr) * 3.6),
                'speed_mode': ('RADIAL' if (self.strict_radial_only or (obj_type in self.disable_unwrap_for_classes)) else 'FUSED'),
                'radial_velocity': float(v_radial),
                'speed_kmh': float(speed_kmh),
                'speed_along_road_kmh': float(speed_along_road_kmh),
                'limit_kmh_used': float(limit_kmh),
                'fused_radial_capped_mps': float(v_rad_fused),
                'speed_along_road_mps': float(abs(v_road)),
                'speed_along_road_signed_mps': float(v_road),
                'road_axis': [float(road_axis[0]), float(road_axis[1]), float(road_axis[2])],
                'cos_los_road': float(cos_lr),
                'angle_los_road_deg': float(angle_los_road_deg),
                'doppler_frequency': float(doppler),
                'track_hits': int(self.hits.get(matched_id, 0)),
                'track_confirmed': True,
                'signal_level': float(signal_level),
                'distance': float(rng),
                'timestamp': float(current_time),
                'direction': direction,
                'motion_state': det.get('motion_state', "unknown"),
            })

            det['score'] = self.calculate_score(det)
            updated_objects.append(det)

        return updated_objects

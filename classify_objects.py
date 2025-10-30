import os
import time
import joblib
import numpy as np
from collections import defaultdict, deque
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from config_utils import load_config

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "radar_lightgbm_model.pkl")

# Keep in sync with trainer
FEATURE_NAMES = [
    "speed_kmh","distance","velocity","signal_level","doppler_frequency",
    "x","y","z","velX","velY","velZ","snr","noise",
    "accX","accY","accZ",
    "range_mean","range_std","noise_mean","noise_std"
]

SAFE = {
    "snr_min": -5.0,     # dB
    "snr_max": 80.0,
    "sig_min": 0.0,      # arbitrary linear
    "sig_max": 120.0,
    "doppler_abs_max": 4000.0,  # Hz
    "speed_kmh_max": 60.0,
    "distance_max": 80.0,
}

def _stats_safe(arr):
    """
    Accepts list/tuple/np.array OR a string like '[1, 2, 3]' or '1,2,3'.
    Returns [mean, std] or [0,0] on failure.
    """
    try:
        if arr is None:
            return [0.0, 0.0]
        if isinstance(arr, str):
            s = arr.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            parts = [p for p in s.split(",") if p.strip() != ""]
            arr = [float(p) for p in parts]
        else:
            arr = [float(x) for x in arr]
        if len(arr) == 0:
            return [0.0, 0.0]
        return [float(np.mean(arr)), float(np.std(arr))]
    except Exception:
        return [0.0, 0.0]

def _g(obj, keys, default=0.0):
    """Get-first helper with tolerant casting."""
    for k in keys:
        if k in obj and obj[k] is not None:
            try:
                return float(obj[k])
            except Exception:
                try:
                    return float(str(obj[k]).strip().replace("dB","").replace("Hz",""))
                except Exception:
                    continue
    return float(default)

class ObjectClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model, self.scaler, self.kept_features = self._load_or_create_model()

        # Map kept features to indices into FEATURE_NAMES
        self._keep_idx = []
        for name in (self.kept_features or []):
            try:
                self._keep_idx.append(FEATURE_NAMES.index(name))
            except ValueError:
                pass
        if not self._keep_idx:
            self._keep_idx = list(range(len(FEATURE_NAMES)))
            self.kept_features = list(FEATURE_NAMES)

        # Remove sklearn verbosity if present (ndarray input is fine)
        try:
            if hasattr(self.model, "feature_names_in_"):
                delattr(self.model, "feature_names_in_")
        except Exception:
            pass

        self._model_mtime = os.path.getmtime(self.model_path) if os.path.exists(self.model_path) else None
        self.object_cache = defaultdict(lambda: deque(maxlen=10))
        self.feature_buffer = deque(maxlen=1000)
        self.config = load_config()
        self.history_cache = {}

    # --------------------------------------------------------------------------------------
    # Model load/fallback
    # --------------------------------------------------------------------------------------
    def _fallback(self):
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, len(FEATURE_NAMES))))
        clf = DummyClassifier(strategy="constant", constant="UNKNOWN")
        clf.fit(np.zeros((1, len(FEATURE_NAMES))), ["UNKNOWN"])
        setattr(clf, "classes_", np.array(["UNKNOWN"]))
        setattr(clf, "feature_name_", FEATURE_NAMES)
        return clf, scaler, list(FEATURE_NAMES)

    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                obj = joblib.load(self.model_path)
                if isinstance(obj, tuple):
                    model, scaler = obj[:2]
                elif isinstance(obj, dict):
                    model, scaler = obj["model"], obj["scaler"]
                else:
                    return self._fallback()

                kept = list(getattr(model, "feature_name_", []) or getattr(model, "feature_names_in_", []) or [])
                if not kept:
                    kept = list(FEATURE_NAMES)

                # Ensure scaler matches kept feature count
                try:
                    n = getattr(scaler, "n_features_in_", None)
                except Exception:
                    n = None
                if (n is not None) and (n != len(kept)):
                    s = StandardScaler()
                    s.fit(np.zeros((1, len(kept))))
                    scaler = s

                if not hasattr(model, "classes_"):
                    setattr(model, "classes_", np.array(["UNKNOWN"]))

                return model, scaler, kept
            except Exception:
                pass
        return self._fallback()

    def reload_if_updated(self):
        try:
            if os.path.exists(self.model_path):
                m = os.path.getmtime(self.model_path)
                if m != self._model_mtime:
                    obj = joblib.load(self.model_path)
                    if isinstance(obj, tuple):
                        model, scaler = obj[:2]
                    elif isinstance(obj, dict):
                        model, scaler = obj["model"], obj["scaler"]
                    else:
                        return

                    kept = list(getattr(model, "feature_name_", []) or getattr(model, "feature_names_in_", []) or []) \
                           or list(FEATURE_NAMES)

                    self.model, self.scaler, self.kept_features = model, scaler, kept
                    try:
                        if hasattr(self.model, "feature_names_in_"):
                            delattr(self.model, "feature_names_in_")
                    except Exception:
                        pass

                    self._keep_idx = [FEATURE_NAMES.index(n) for n in self.kept_features if n in FEATURE_NAMES] \
                                     or list(range(len(FEATURE_NAMES)))

                    self._model_mtime = m
        except Exception:
            pass  # keep old model if reload fails

    # --------------------------------------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------------------------------------
    def _extract_features(self, obj: dict):
        # Tolerant to naming from TLVs/fusion
        speed_kmh = _g(obj, ["speed_kmh", "speedKmh", "speed"], 0.0)
        distance  = _g(obj, ["radar_distance", "distance", "range"], 0.0)
        velocity  = abs(_g(obj, ["velocity", "radial_velocity", "speed_mps"], 0.0))
        signal    = _g(obj, ["signal_level", "signal"], 0.0)
        doppler   = _g(obj, ["doppler_frequency", "doppler"], 0.0)

        x = _g(obj, ["x", "posX"], 0.0)
        y = _g(obj, ["y", "posY"], 0.0)
        z = _g(obj, ["z", "posZ"], 0.0)

        velX = _g(obj, ["velX", "velx", "vx"], 0.0)
        velY = _g(obj, ["velY", "vely", "vy"], 0.0)
        velZ = _g(obj, ["velZ", "velz", "vz"], 0.0)

        snr   = _g(obj, ["snr", "SNR"], 0.0)
        noise = _g(obj, ["noise"], 0.0)

        accX = _g(obj, ["accX", "accx", "ax"], 0.0)
        accY = _g(obj, ["accY", "accy", "ay"], 0.0)
        accZ = _g(obj, ["accZ", "accz", "az"], 0.0)

        r_mean, r_std = _stats_safe(obj.get("range_profile"))
        n_mean, n_std = _stats_safe(obj.get("noise_profile"))

        if not np.isfinite(speed_kmh) or speed_kmh < 0 or speed_kmh > SAFE["speed_kmh_max"]:
            speed_kmh = max(0.0, min(abs(velocity) * 3.6, SAFE["speed_kmh_max"]))
        if not np.isfinite(distance) or distance < 0 or distance > SAFE["distance_max"]:
            distance = max(0.0, min(distance if np.isfinite(distance) else 0.0, SAFE["distance_max"]))
        if not np.isfinite(signal) or signal < SAFE["sig_min"] or signal > SAFE["sig_max"]:
            signal = max(SAFE["sig_min"], min(signal if np.isfinite(signal) else 0.0, SAFE["sig_max"]))
        if not np.isfinite(snr) or snr < SAFE["snr_min"] or snr > SAFE["snr_max"]:
            snr = max(SAFE["snr_min"], min(snr if np.isfinite(snr) else 0.0, SAFE["snr_max"]))
        if not np.isfinite(doppler) or abs(doppler) > SAFE["doppler_abs_max"]:
            doppler = 0.0

        # Backfill canonical keys so downstream sees them
        obj.setdefault("x", x)
        obj.setdefault("y", y)
        obj.setdefault("z", z)
        obj.setdefault("radar_distance", distance)
        obj.setdefault("snr", snr)
        obj.setdefault("signal_level", signal)
        obj.setdefault("doppler_frequency", doppler)
        obj.setdefault("speed_kmh", speed_kmh)

        return [
            speed_kmh, distance, velocity, signal, doppler,
            x, y, z, velX, velY, velZ, snr, noise,
            accX, accY, accZ,
            r_mean, r_std, n_mean, n_std
        ]

    # --------------------------------------------------------------------------------------
    # Heuristics & smoothing
    # --------------------------------------------------------------------------------------
    def _track_key(self, obj: dict):
        return obj.get("object_id") or obj.get("source_id") or obj.get("id")

    def _human_vehicle_rule(self, obj: dict, feats: list) -> tuple[str, float, dict]:
        """
        Rule-based override resilient to bad models:
        - HUMAN if slow & close (gate scenario) with moderate SNR/signal
        - VEHICLE if fast OR very high SNR close-by
        Returns (maybe_label, boost, debug)
        """
        cfg = (self.config or {}).get("classification", {}) or {}
        # Defaults tuned for your site; editable in config.json under "classification"
        HUMAN_V_MAX = float(cfg.get("human_speed_hi_kmh", 9.0))
        HUMAN_D_MAX = float(cfg.get("human_dist_hi_m", 12.0))
        HUMAN_D_MIN = float(cfg.get("human_dist_lo_m", 0.3))
        VEH_SPEED   = float(cfg.get("vehicle_speed_lo_kmh", 12.0))
        VEH_SNR_HI  = float(cfg.get("vehicle_snr_hi_db", 35.0))

        speed = float(obj.get("speed_kmh", feats[0] if feats else 0.0))
        dist  = float(obj.get("radar_distance", obj.get("distance", feats[1] if feats else 0.0)))
        snr   = float(obj.get("snr", feats[11] if feats else 0.0))
        sig   = float(obj.get("signal_level", feats[3] if feats else 0.0))
        vx    = float(obj.get("velX", 0.0)); vy = float(obj.get("velY", 0.0))
        vel_jitter = 0.0

        # Use short velocity history jitter as a proxy for gait (humans are "twitchier" laterally)
        k = self._track_key(obj)
        if k:
            hist = self.object_cache[k]
            if len(hist) >= 3:
                vxs = [h[0] for h in hist]; vys = [h[1] for h in hist]
                vel_jitter = float(np.std(vxs) + np.std(vys))
        debug = {"speed": speed, "dist": dist, "snr": snr, "sig": sig, "jitter": vel_jitter}

        # Primary HUMAN rule: slow & near the sensor (typical person walking near the gate)
        if HUMAN_D_MIN <= dist <= HUMAN_D_MAX and speed <= HUMAN_V_MAX and snr < VEH_SNR_HI:
            return "HUMAN", 0.15, {**debug, "rule": "human_slow_close"}

        # VEHICLE rule: clearly fast OR very high SNR at close range
        if speed >= VEH_SPEED or (dist <= 10.0 and snr >= VEH_SNR_HI):
            return "VEHICLE", 0.10, {**debug, "rule": "vehicle_fast_or_strong"}

        # Gait fallback: if jittery and not fast, bias to HUMAN
        if vel_jitter >= 0.25 and speed <= 10.0:
            return "HUMAN", 0.05, {**debug, "rule": "human_jitter"}

        return "UNKNOWN", 0.0, {**debug, "rule": "none"}

    def _smooth_label(self, obj_key, proposed: str, conf: float) -> tuple[str, float]:
        dq = self.history_cache.setdefault(obj_key, deque(maxlen=6))
        dq.append((proposed, float(conf)))
        # Majority with confidence weighting
        tallies = defaultdict(float)
        for lbl, c in dq:
            tallies[lbl] += max(0.5, float(c))  # keep at least 0.5 weight
        winner = max(tallies.items(), key=lambda kv: kv[1])[0]
        # Boost smoothed confidence if consistent
        consistency = tallies[winner] / max(1.0, sum(tallies.values()))
        return winner, float(min(0.99, max(conf, 0.5 + 0.4 * consistency)))

    # --------------------------------------------------------------------------------------
    # Heuristic fallback when the model is UNKNOWN-only or predicts UNKNOWN
    # --------------------------------------------------------------------------------------
    def _coerce_unknown(self, obj: dict) -> str:
        return "Object"

    # --------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------
    def classify_objects(self, objects):
        # Hot-reload if file changed
        self.reload_if_updated()

        out = []
        for obj in objects:
            feats_full = self._extract_features(obj)
            full = np.asarray([feats_full], dtype=float)
            X = full[:, self._keep_idx]

            # Scale
            try:
                Xs = self.scaler.transform(X)
            except Exception:
                # rebuild neutral scaler to keep pipeline alive
                self.model, self.scaler, self.kept_features = self._fallback()
                self._keep_idx = list(range(len(FEATURE_NAMES)))
                Xs = self.scaler.transform(full[:, self._keep_idx])

            # Predict
            try:
                probs = self.model.predict_proba(Xs)[0]
                classes = list(self.model.classes_)
                k = int(np.argmax(probs)) if len(probs) else 0
                label = classes[k] if 0 <= k < len(classes) else "UNKNOWN"
                conf  = float(probs[k]) if len(probs) else 0.0
                raw = {c: float(p) for c, p in zip(classes, probs)}
            except Exception:
                classes = list(getattr(self.model, "classes_", []) or [])
                label = classes[0] if classes else "UNKNOWN"
                conf, raw = 1.0, {}

            # If the model is UNKNOWN-only or predicted UNKNOWN → coerce
            classes = list(getattr(self.model, "classes_", []) or [])
            if (len(classes) == 1 and classes[0] == "UNKNOWN") or label == "UNKNOWN":
                label = self._coerce_unknown(obj)
                conf = min(conf, 0.51)

            # ---------- Robust HUMAN/VEHICLE guardrails ----------
            rule_label, boost, dbg = self._human_vehicle_rule(obj, feats_full)
            pre_rule = label
            if rule_label in ("HUMAN", "VEHICLE"):
                # Only override if model is weak OR disagrees with heuristic in low-speed regimes
                if (conf < 0.7) or (rule_label == "HUMAN" and label == "VEHICLE"):
                    label = rule_label
                    conf = float(min(0.98, max(conf, 0.65 + boost)))

            # ---------- Per-track smoothing ----------
            key = self._track_key(obj)
            if key:
                label, conf = self._smooth_label(key, label, conf)
                # cache velocity history for jitter calc
                vx = float(obj.get("velX", obj.get("vx", 0.0)) or 0.0)
                vy = float(obj.get("velY", obj.get("vy", 0.0)) or 0.0)
                self.object_cache[key].append((vx, vy, time.time()))

            # Motion state + direction (uses dynamic limits from config)
            cfg = self.config or {}
            limits = cfg.get("dynamic_speed_limits", {})
            default_limit = float(limits.get("default", 5.0))
            speed_limit = float(limits.get(label.upper(), default_limit))

            speed_kmh = _g(obj, ["speed_kmh", "speedKmh", "speed"], feats_full[0])
            vel_x = _g(obj, ["velX", "vx"], feats_full[8])
            vel_y = _g(obj, ["velY", "vy"], feats_full[9])

            if float(speed_kmh) >= speed_limit:
                motion = "SPEEDING"
            elif float(speed_kmh) > 0.3:
                motion = "MOVING"
            else:
                motion = "STATIONARY"

            if abs(float(vel_y)) < 0.05 and abs(float(vel_x)) < 0.05:
                direction = "STATIC"
            elif abs(float(vel_y)) >= abs(float(vel_x)):
                direction = "AWAY" if float(vel_y) > 0 else "TOWARDS"
            else:
                direction = "RIGHT" if float(vel_x) > 0 else "LEFT"

            obj_out = dict(obj)
            obj_out.update({
                "type": label,
                "confidence": round(float(conf or 0.0), 3),
                "raw_probabilities": raw,
                "score": round((conf or 0.0) * float(obj.get("snr", feats_full[11] or 1.0)), 2),
                "motion_state": motion,
                "direction": direction,
                "debug_class": {
                    "pre_rule": pre_rule,
                    "rule": (dbg or {}).get("rule", "none"),
                    "details": {k: v for k, v in (dbg or {}).items() if k != "rule"}
                }
            })
            out.append(obj_out)

        return out

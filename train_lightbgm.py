import os
import sys
import psycopg2
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from collections import Counter
import tempfile
import time
import shutil

# --- Headless matplotlib (no GUI on the Pi) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "radar_lightgbm_model.pkl")

DB_CONFIG = {
    "dbname": "iwr6843_db",
    "user": "radar_user",
    "password": "securepass123",
    "host": "localhost"
}

FEATURE_NAMES = [
    "speed_kmh", "distance", "velocity", "signal_level", "doppler_frequency",
    "x", "y", "z", "velX", "velY", "velZ", "snr", "noise",
    "accX", "accY", "accZ",
    "range_mean", "range_std", "noise_mean", "noise_std"
]

# Canonicalize noisy labels coming from pipeline/annotation
CANON_MAP = {
    # PERSON-like
    "PERSON": "HUMAN", "PEDESTRIAN": "HUMAN", "HUMAN": "HUMAN", "MAN": "HUMAN", "WOMAN": "HUMAN",
    # VEHICLE-like (treat RC/Hot Wheels as vehicle)
    "CAR": "VEHICLE", "TRUCK": "VEHICLE", "BUS": "VEHICLE", "BIKE": "VEHICLE",
    "BICYCLE": "VEHICLE", "MOTORBIKE": "VEHICLE", "MOTORCYCLE": "VEHICLE", "AUTO": "VEHICLE",
    "SCOOTER": "VEHICLE", "RC_CAR": "VEHICLE", "VEHICLE": "VEHICLE",
}

def _canon_label(raw):
    s = (raw or "").strip().upper()
    return CANON_MAP.get(s, "UNKNOWN")

def fetch_training_data():
    query = """
        SELECT speed_kmh, distance, velocity, signal_level, doppler_frequency,
               x, y, z, velX, velY, velZ, snr, noise,
               accx, accy, accz,
               range_profile, noise_profile,
               type
        FROM radar_data
        WHERE type IS NOT NULL AND TRIM(type) <> ''
          AND speed_kmh IS NOT NULL AND distance IS NOT NULL
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[ERROR] Failed to fetch training data: {e}")
        return []

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

def _abort_if_all_unknown(y_array):
    """
    Refuse to overwrite the current model if labels are empty or all UNKNOWN.
    Return True if we should abort training.
    """
    try:
        y = np.asarray(y_array, dtype=str)
        if y.size == 0:
            print("[ABORT] No training rows; refusing to overwrite the model.")
            return True
        uniq = {s.strip().upper() for s in y.tolist()}
        if uniq == {"UNKNOWN"}:
            print("[ABORT] All labels are UNKNOWN; refusing to overwrite the model.")
            return True
    except Exception as e:
        print(f"[WARN] Guard check failed: {e}")
    return False

def train_and_save_model(rows):
    X, y = [], []

    for row in rows:
        *base, accx, accy, accz, range_profile, noise_profile, label = row
        # base = [speed_kmh, distance, velocity, signal_level, doppler_frequency,
        #         x, y, z, velX, velY, velZ, snr, noise]
        base[2] = abs(base[2]) if base[2] is not None else 0.0  # |velocity|

        acc_feats = [accx or 0.0, accy or 0.0, accz or 0.0]
        rng_mean, rng_std = _stats_safe(range_profile)
        noi_mean, noi_std = _stats_safe(noise_profile)

        feats = [(f if f is not None else 0.0) for f in base] + acc_feats + [rng_mean, rng_std, noi_mean, noi_std]
        X.append(feats)
        y.append(_canon_label(label))

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=str)

    print(f"[DEBUG] Label distribution (raw): {Counter(y)}")

    # If any non-UNKNOWN labels exist, drop UNKNOWN rows to avoid bias
    non_unknown_mask = (y != "UNKNOWN")
    if np.any(non_unknown_mask):
        X = X[non_unknown_mask]
        y = y[non_unknown_mask]
        print(f"[INFO] Dropped UNKNOWN rows; new distribution: {Counter(y)}")

    if _abort_if_all_unknown(y):
        return False

    # --- Drop zero-variance columns BEFORE scaling/training --------------------
    var = np.nanvar(X, axis=0)
    keep_mask = var > 1e-12
    dropped = [name for name, keep in zip(FEATURE_NAMES, keep_mask) if not keep]
    kept    = [name for name, keep in zip(FEATURE_NAMES, keep_mask) if keep]

    if dropped:
        print(f"[INFO] Dropping {len(dropped)} constant feature(s): {', '.join(dropped)}")
    else:
        print("[INFO] No constant features detected.")

    X = X[:, keep_mask]
    feature_names_used = kept

    # Quick coverage report to explain importances later
    nz_ratio = (np.count_nonzero(X, axis=0) / max(1, X.shape[0]))
    print("[COVERAGE] non-zero ratio per kept feature:")
    for n, r in zip(feature_names_used, nz_ratio):
        print(f"  - {n}: {r:.2f}")

    # --- Scale + Train --------------------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    labels_set = set(y)
    if len(labels_set) == 1:
        only = next(iter(labels_set))
        print(f"[WARN] Single-class training: {only}")
        model = DummyClassifier(strategy="constant", constant=only)
        model.fit(Xs, y)
        acc = float(model.score(Xs, y))
        print(f"[INFO] ACCURACY: {acc*100:.2f}%")
    else:
        Xtr, Xval, ytr, yval = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(labels_set),
            learning_rate=0.05,
            n_estimators=200,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=0)]
        )
        acc = model.score(Xval, yval)
        print(f"[INFO] ACCURACY: {acc*100:.2f}%")

    # Attach the *actual* feature list used
    try:
        setattr(model, "feature_name_", feature_names_used)
    except Exception:
        pass

    save_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(save_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="radar_model_", suffix=".pkl", dir=save_dir)
    os.close(tmp_fd)
    try:
        artifact = {
            "model": model,
            "scaler": scaler,
            "features": list(feature_names_used),
            "labels": sorted(list(set(y.tolist())))
        }
        joblib.dump(artifact, tmp_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    def _replace_with_retries(src, dst, attempts=6, delay=0.25):
        """
        Retry os.replace for transient Windows sharing violations (AV/indexer).
        """
        last = None
        for i in range(attempts):
            try:
                os.replace(src, dst)  # atomic on NTFS
                return True
            except Exception as e:
                last = e
                time.sleep(delay * (1.6 ** i))
        print(f"[WARN] os.replace failed after retries: {last}")
        return False

    if not _replace_with_retries(tmp_path, MODEL_PATH):
        # Fallback: copy then remove temp
        shutil.copyfile(tmp_path, MODEL_PATH)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")
    # Emit a parseable accuracy line for the retrain API
    if acc is not None:
        # normalized (0.0–1.0) for easy parsing, *in addition* to the % line above
        print(f"accuracy={acc:.4f}")

    # --- Feature-importance plot for the kept features ------------------------
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None and hasattr(model, "booster_"):
            importances = model.booster_.feature_importance()

        if importances is not None and len(importances) == len(feature_names_used):
            order = np.argsort(importances)  # sort for readability
            plt.figure(figsize=(10, 6))
            plt.barh([feature_names_used[i] for i in order], np.array(importances)[order])
            plt.title("LightGBM Feature Importance")
            plt.xlabel("Importance Score")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            print("[INFO] Saved feature importance to feature_importance.png")
        else:
            print("[SKIP] Feature importance skipped (no importances or length mismatch).")
    except Exception as e:
        print(f"[WARN] Skipped feature importance plot: {e}")

    return True, acc, list(feature_names_used), sorted(list(set(y.tolist())))

def main():
    rows = fetch_training_data()
    total = len(rows)
    print(f"[INFO] Fetched {total} samples from DB")
    if total == 0:
        print("[ABORT] No training rows; refusing to overwrite the model.")
        print("ABORT_REASON: NO_ROWS")
        raise SystemExit(1)

    ok, acc, feats, labels = train_and_save_model(rows)
    if not ok:
        print("ABORT_REASON: ALL_UNKNOWN_OR_SINGLE_CLASS")
        raise SystemExit(1)
    if acc is not None:
        print(f"[META] features={len(feats)} labels={','.join(labels)}")

if __name__ == "__main__":
    main()

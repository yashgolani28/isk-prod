#!/usr/bin/env python3
from iwr6843_interface import IWR6843Interface
from radar_logger import IWR6843Logger
from classify_objects import ObjectClassifier
from config_utils import load_config

import time
import socket
import json
import numpy as np
from sklearn.cluster import DBSCAN

# ---- setup -------------------------------------------------------------------
classifier = ObjectClassifier()
cfg = load_config()

ports = cfg.get("iwr_ports", {})
cli_port = ports.get("cli", "/dev/ttyUSB0")  # not used here, interface handles
data_port = ports.get("data", "/dev/ttyUSB3")
cfg_path = ports.get("cfg_path", "oob_isk.cfg")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DEST = ("127.0.0.1", 5005)

radar = IWR6843Interface()
logger = IWR6843Logger()

print("[INFO] Started reading radar data... Press Ctrl+C to stop.")

# ---- helpers -----------------------------------------------------------------
def _extract_xyz_doppler(points):
    """
    Accepts either:
      - list[dict] with keys: x,y,z,doppler,(snr),(noise)
      - list[list/tuple] columns: [x,y,z,doppler,(snr),(noise),(trackIdx?)]
    Returns: XYZ (N,3), D (N,), SNR (N,), NOISE (N,)
    """
    # Handle None / empty without ambiguous truth tests on numpy arrays
    if points is None:
        return (np.empty((0,3), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32))
    if isinstance(points, np.ndarray):
        if points.size == 0:
            return (np.empty((0,3), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32),
                    np.empty((0,), np.float32))
        A = points.astype(np.float32, copy=False)
        XYZ = A[:, :3]
        D   = A[:, 3] if A.shape[1] > 3 else np.zeros((A.shape[0],), np.float32)
        S   = A[:, 4] if A.shape[1] > 4 else np.zeros((A.shape[0],), np.float32)
        N   = A[:, 5] if A.shape[1] > 5 else np.zeros((A.shape[0],), np.float32)
        return XYZ, D, S, N
    if hasattr(points, "__len__") and len(points) == 0:
        return (np.empty((0,3), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32))
    first = points[0]
    if isinstance(first, dict):
        XYZ = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in points], np.float32)
        D   = np.array([p.get("doppler", 0.0) for p in points], np.float32)
        S   = np.array([p.get("snr", 0.0) for p in points], np.float32)
        N   = np.array([p.get("noise", 0.0) for p in points], np.float32)
        return XYZ, D, S, N
    A = np.asarray(points, dtype=np.float32)
    XYZ = A[:, :3]
    D   = A[:, 3] if A.shape[1] > 3 else np.zeros((A.shape[0],), np.float32)
    S   = A[:, 4] if A.shape[1] > 4 else np.zeros((A.shape[0],), np.float32)
    N   = A[:, 5] if A.shape[1] > 5 else np.zeros((A.shape[0],), np.float32)
    return XYZ, D, S, N

def derive_targets_from_pointcloud(points):
    """
    Small DBSCAN-based object builder for OOB frames without trackData.
    Produces 'track-like' dicts compatible with downstream pipeline.
    """
    XYZ, D, S, N = _extract_xyz_doppler(points)
    if XYZ.shape[0] == 0:
        return []

    # Tune eps/min_samples per site; start conservative
    labels = DBSCAN(eps=0.45, min_samples=5).fit_predict(XYZ)

    targets = []
    cid = 1
    for k in set(labels):
        if k == -1:
            continue
        m = labels == k
        C = XYZ[m]
        pos = C.mean(axis=0)

        # Rough radial speed from doppler: v ≈ (λ/2) * fd, λ≈5 mm at 60 GHz
        v_rad = float(D[m].mean()) * 0.005 / 2.0 if D.size else 0.0

        targets.append({
            "id": cid,
            "posX": float(pos[0]), "posY": float(pos[1]), "posZ": float(pos[2]),
            "velX": 0.0, "velY": v_rad, "velZ": 0.0,
            "snr": float(np.median(S[m])) if S.size else 0.0,
            "noise": float(np.median(N[m])) if N.size else 0.0,
            "speed_kmh": abs(v_rad) * 3.6,
            "distance": float(np.linalg.norm(pos)),
        })
        cid += 1

    return targets

# ---- main loop ---------------------------------------------------------------
heartbeat_counter = 0
try:
    while True:
        frame = radar.get_targets()

        # OOB may not publish tracker targets
        targets = frame.get("trackData", [])
        # Coalesce without using truthiness on numpy arrays
        points = frame.get("pointCloud", None)
        if points is None:
            points = frame.get("point_cloud", None)
        if points is None:
            points = []

        # Prefer OOB range-azimuth heatmap key when present (not used here, but read for debug)
        targets = frame.get("trackData", [])
        # Coalesce without using truthiness on numpy arrays
        points = frame.get("pointCloud", None)
        if points is None:
            points = frame.get("point_cloud", None)
        if points is None:
            points = []

        # Quick debug: show points count (lets us know if CFAR/cfg is the issue)
        # Robust length for list or ndarray
        num_points = (points.shape[0] if isinstance(points, np.ndarray)
                      else (len(points) if hasattr(points, "__len__") else 0))
        print(f"[Frame @ {time.strftime('%H:%M:%S')}] points={num_points}", end="")

        # If no tracker list, build objects from point cloud
        if not targets and points:
            targets = derive_targets_from_pointcloud(points)

        # Classify for type/confidence; safe if empty
        classified = classifier.classify_objects(targets) if targets else []

        # Send classified targets over UDP (same wire format as main)
        sock.sendto(json.dumps({"targets": classified}).encode("utf-8"), DEST)

        # Heartbeat (approx every second at 0.1 s loop)
        heartbeat_counter += 1
        if heartbeat_counter >= 10:
            sock.sendto(json.dumps({"status": "radar_alive"}).encode("utf-8"), DEST)
            heartbeat_counter = 0

        # Print summary + per-object details (safe defaults)
        print(f" | targets={len(classified)}")
        for obj in classified:
            # Ensure AoA exists for display
            if "azimuth" not in obj or "elevation" not in obj:
                x, y, z = obj.get("posX", 0.0), obj.get("posY", 0.0), obj.get("posZ", 0.0)
                obj["azimuth"] = round(np.degrees(np.arctan2(x, y)), 2)
                obj["elevation"] = round(np.degrees(np.arctan2(z, np.hypot(x, y))), 2)

            obj_id = obj.get("id", obj.get("object_id", "NA"))

            print(
                f"ID: {obj_id!s:>2} | "
                f"Type: {obj.get('type','NA'):<8} | "
                f"Speed: {obj.get('speed_kmh',0.0):>6.2f} km/h | Dist: {obj.get('distance',0.0):>5.2f} m | "
                f"Conf: {obj.get('confidence',0.0):.2f} | G: {obj.get('g',0.0):.2f}\n"
                f" ↳ Pos [X:{obj.get('posX',0.0):>5.2f} Y:{obj.get('posY',0.0):>5.2f} Z:{obj.get('posZ',0.0):>5.2f}] m | "
                f"Vel [X:{obj.get('velX',0.0):>5.2f} Y:{obj.get('velY',0.0):>5.2f} Z:{obj.get('velZ',0.0):>5.2f}] m/s | "
                f"Acc [X:{obj.get('accX',0.0):>5.2f} Y:{obj.get('accY',0.0):>5.2f} Z:{obj.get('accZ',0.0):>5.2f}] m/s²\n"
                f" ↳ AoA → Azimuth: {obj['azimuth']:>6.2f}° | Elevation: {obj['elevation']:>6.2f}°"
            )

        # Log once per frame
        if classified:
            logger.log_targets(classified)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Stopped radar read loop.")

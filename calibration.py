from __future__ import annotations
import os, json, time, glob, re
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import cv2

from projection import (Intrinsics, Extrinsics, CameraModel,
                        radar_spherical_to_cart, solve_extrinsics_pnp)

# --- paths anchored to this module directory ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CAL_DIR    = os.path.join(BASE_DIR, "calibration")
PAIRS_PATH = os.path.join(CAL_DIR, "calibration_pairs.json")
MODEL_PATH = os.path.join(CAL_DIR, "camera_model.json")
STAGING_PATH = os.path.join(CAL_DIR, "camera_model_staging.json")

def _paths_for_camera(camera_id: Optional[str]):
    """
    Canonical per-camera path resolver used by the UI (app.py).
    Returns: (base_dir, pairs_path, live_path, staged_path)

    - When camera_id is None → legacy global paths under calibration/
    - When camera_id is set  → per-camera paths under calibration/cams/<slug>/
      (matches _model_paths_for and where publish_model() writes)
    """
    _ensure_dir()
    slug = _slug_camera_id(camera_id)
    if not slug:
        # legacy single-camera layout
        return CAL_DIR, PAIRS_PATH, MODEL_PATH, STAGING_PATH

    cam_dir = os.path.join(CAL_DIR, "cams", slug)   # <-- unified 'cams' folder
    os.makedirs(cam_dir, exist_ok=True)

    pairs_path   = _pairs_path_for(camera_id)       # calibration/pairs/<slug>.json
    live_path    = os.path.join(cam_dir, "camera_model.json")
    staged_path  = os.path.join(cam_dir, "camera_model_staging.json")
    return cam_dir, pairs_path, live_path, staged_path

def _ensure_dir(dirpath: Optional[str]=None):
    os.makedirs(dirpath or CAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(CAL_DIR, "pairs"), exist_ok=True)
    os.makedirs(os.path.join(CAL_DIR, "cams"), exist_ok=True)
def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _slug_camera_id(camera_id: Optional[str]) -> Optional[str]:
    """
    Compact, filesystem-safe slug for a camera identifier (id/name).
    Returns None if camera_id is falsy.
    """
    if not camera_id:
        return None
    s = str(camera_id).strip()
    # keep alnum, dash/underscore only
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    return s or None

def _pairs_path_for(camera_id: Optional[str]) -> str:
    """
    Per-camera pairs location. Falls back to legacy global path when camera_id is None.
    """
    _ensure_dir()
    slug = _slug_camera_id(camera_id)
    if not slug:
        return PAIRS_PATH  # legacy global
    return os.path.join(CAL_DIR, "pairs", f"{slug}.json")

def _model_paths_for(camera_id: Optional[str]) -> Dict[str, str]:
    """
    Returns dict with: live, staged, archive_dir, camera_dir.
    For legacy (None), returns the global files.
    """
    _ensure_dir()
    slug = _slug_camera_id(camera_id)
    if not slug:
        return {
            "live": MODEL_PATH,
            "staged": STAGING_PATH,
            "archive_dir": CAL_DIR,
            "camera_dir": CAL_DIR,
        }
    cam_dir = os.path.join(CAL_DIR, "cams", slug)
    os.makedirs(cam_dir, exist_ok=True)
    return {
        "live": os.path.join(cam_dir, "camera_model.json"),
        "staged": os.path.join(cam_dir, "camera_model_staging.json"),
        "archive_dir": cam_dir,
        "camera_dir": cam_dir,
    }

# ---------- Pair I/O ----------
def load_pairs(camera_id: Optional[str] = None, *, all_cameras: bool = False) -> List[Dict]:
    """
    Load calibration pairs.
    - If camera_id is provided, load only that camera's pairs.
    - If all_cameras=True, load & merge all per-camera files (plus legacy global if present).
    - Else (default), load legacy global file (backward-compatible).
    """
    _ensure_dir()
    if all_cameras:
        pairs: List[Dict] = []
        # include legacy if it exists
        if os.path.exists(PAIRS_PATH):
            try:
                with open(PAIRS_PATH, "r") as f:
                    pairs.extend(json.load(f) or [])
            except Exception:
                pass
        # include all per-camera files
        pdir = os.path.join(CAL_DIR, "pairs")
        for name in sorted(os.listdir(pdir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(pdir, name), "r") as f:
                    ps = json.load(f) or []
                    # tag with source cam if missing
                    cam = name[:-5]
                    for p in ps:
                        p.setdefault("meta", {})
                        p["meta"].setdefault("camera_id", cam)
                    pairs.extend(ps)
            except Exception:
                continue
        return pairs
    # single file (per-camera or legacy)
    path = _pairs_path_for(camera_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f) or []

def save_pairs(pairs: List[Dict], camera_id: Optional[str] = None):
    """
    Save pairs for a specific camera (or legacy global when camera_id=None).
    """
    _ensure_dir()
    path = _pairs_path_for(camera_id)
    with open(path, "w") as f:
        json.dump(pairs or [], f, indent=2)

def reset_pairs(camera_id: Optional[str] = None, *, all_cameras: bool = False) -> None:
    """
    Clear collected calibration pairs.
    - camera_id=None and all_cameras=False → reset legacy global only
    - camera_id=<id> → reset only that camera
    - all_cameras=True → reset ALL per-camera files + legacy
    """
    _ensure_dir()
    targets: Iterable[str]
    if all_cameras:
        targets = [PAIRS_PATH] + [
            os.path.join(CAL_DIR, "pairs", n)
            for n in os.listdir(os.path.join(CAL_DIR, "pairs"))
            if n.endswith(".json")
        ]
    else:
        targets = [_pairs_path_for(camera_id)]
    for p in targets:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        # recreate as empty list
        try:
            with open(p, "w") as f:
                json.dump([], f, indent=2)
        except Exception:
            pass

# ---------- Public API ----------
def add_pair(u: float, v: float, r_m: float, az_deg: float, el_deg: float,
             meta: Optional[Dict]=None, camera_id: Optional[str] = None):
    """
    Record one correspondence from a detection:
      - (u,v): pixel center of matched bbox
      - (r, az, el): radar spherical
    """
    pairs = load_pairs(camera_id=camera_id)
    rec = {
        "ts": _now_iso(),
        "u": float(u), "v": float(v),
        "r": float(r_m), "az": float(az_deg), "el": float(el_deg),
        "meta": (meta or {}).copy()
    }
    # stamp camera_id into meta
    rec["meta"].setdefault("camera_id", _slug_camera_id(camera_id))
    pairs.append(rec)
    save_pairs(pairs, camera_id=camera_id)

def estimate_intrinsics(res_w: int, res_h: int, fov_h_deg: float,
                        distortion: Optional[Tuple[float,float,float,float,float]] = None) -> Intrinsics:
    intr = Intrinsics.from_fov(res_w, res_h, fov_h_deg)
    if distortion is not None:
        intr.dist = distortion
    return intr

def fit_extrinsics(intr: Intrinsics, min_points: int = 10,
                   camera_id: Optional[str] = None) -> Tuple[CameraModel, float, int]:
    """
    Fit extrinsics for a specific camera (or legacy global when camera_id=None).
    Writes a per-camera *staged* model; require publish_model(camera_id) to make it live.
    """
    pairs = load_pairs(camera_id=camera_id)
    if len(pairs) < max(6, min_points):
        raise RuntimeError(f"Need at least {max(6,min_points)} pairs, have {len(pairs)}")

    # Build arrays
    img = np.array([[p["u"], p["v"]] for p in pairs], dtype=np.float64)
    world = np.array([radar_spherical_to_cart(p["r"], p["az"], p["el"]).reshape(3,)
                      for p in pairs], dtype=np.float64)

    # Normalize: remove obvious outliers by percentile
    # (Keep 5-95% in range to help PnP)
    # We’ll rely on RANSAC inside solve_extrinsics_pnp too.
    extr, med_err = solve_extrinsics_pnp(world, img, intr, use_ransac=True)

    model = CameraModel(intr=intr, extr=extr, meta={
        "created": _now_iso(),
        "pairs_used": len(pairs),
        "median_reproj_error_px": med_err,
        "source": "radar-visual PnP (3DPC)",
        "notes": "PnP with RANSAC; intrinsics from FOV",
        "camera_id": _slug_camera_id(camera_id)
    })
    # stage first — require publish() to go live (per-camera)
    paths = _model_paths_for(camera_id)
    _ensure_dir()
    with open(paths["staged"], "w") as f:
        json.dump({
            "intr": {
                "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "dist": list(intr.dist)
            },
            "extr": {"R_rc": extr.R_rc, "t_rc": extr.t_rc},
            "meta": model.meta
        }, f, indent=2)
    return model, med_err, len(pairs)

def publish_model(camera_id: Optional[str] = None, *, make_active_global: bool = True) -> str:
    """
    Promote the per-camera staged model to its live file and archive it.
    If make_active_global=True, also copy to the legacy global live path so the
    existing pipeline keeps reading calibration/camera_model.json for the *selected* camera.
    Returns the archived version path.
    """
    paths = _model_paths_for(camera_id)
    staged = paths["staged"]
    if not os.path.exists(staged):
        raise RuntimeError("Nothing staged. Run fit_extrinsics() first.")
    _ensure_dir()
    # version the file inside this camera's dir
    versioned = os.path.join(paths["camera_dir"], f"camera_model_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(staged, "r") as f:
        data = json.load(f)
    # write per-camera live
    with open(paths["live"], "w") as f:
        json.dump(data, f, indent=2)
    # archive
    with open(versioned, "w") as f:
        json.dump(data, f, indent=2)
    # optionally mirror to global live for backward-compat
    if make_active_global:
        with open(MODEL_PATH, "w") as f:
            json.dump(data, f, indent=2)
    return versioned

def validate_on_samples(intr: Intrinsics, sample_pairs: Optional[List[Dict]]=None,
                        camera_id: Optional[str]=None) -> float:
    """
    Reprojection median error on provided samples or current pairs.
    """
    pairs = sample_pairs or load_pairs(camera_id=camera_id)
    if len(pairs) < 6:
        return float("inf")
    img = np.array([[p["u"], p["v"]] for p in pairs], dtype=np.float64)
    world = np.array([radar_spherical_to_cart(p["r"], p["az"], p["el"]).reshape(3,)
                      for p in pairs], dtype=np.float64)

    # Load staged/live model (prefer staged)
    _, _, live_path, staging_path = _paths_for_camera(camera_id)
    path = staging_path if os.path.exists(staging_path) else live_path
    m = CameraModel.load(path)
    # Build project with cv2 to avoid drift
    K = m.intr.K(); dist = m.intr.dist_coeffs()
    R, t = np.array(m.extr.R_rc, np.float64), np.array(m.extr.t_rc, np.float64).reshape(3,1)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(world, rvec, t, K, dist)
    err = np.linalg.norm(proj.reshape(-1,2) - img, axis=1)
    return float(np.median(err))

def cameras_with_pairs() -> List[str]:
    """
    Utility for UI: list camera slugs that have any pair file.
    """
    _ensure_dir()
    out = []
    pdir = os.path.join(CAL_DIR, "pairs")
    for n in sorted(os.listdir(pdir)):
        if n.endswith(".json"):
            try:
                fp = os.path.join(pdir, n)
                if os.path.getsize(fp) > 2:
                    out.append(n[:-5])
            except Exception:
                continue
    # include legacy marker if old global file exists
    if os.path.exists(PAIRS_PATH) and os.path.getsize(PAIRS_PATH) > 2:
        out.append("legacy")
    return out

def drift_check(live_samples: List[Dict], threshold_px: float = 8.0,
                camera_id: Optional[str]=None) -> Dict:
    """
    Use last-N live matched detections to assess drift.
    Return {median_error_px, status}
    """
    # Compose Intrinsics from live model
    _, _, live_path, _ = _paths_for_camera(camera_id)
    m = CameraModel.load(live_path)
    err = validate_on_samples(m.intr, sample_pairs=live_samples, camera_id=camera_id)
    return {"median_error_px": err, "status": "OK" if err <= threshold_px else "RECALIBRATE"}

# ---------- Maintenance helpers ----------
def delete_model(delete_staging: bool = True, camera_id: Optional[str]=None) -> Dict[str, bool]:
    """
    Remove live camera model (and optional staged model).
    Returns a map of path->bool indicating if a file was removed.
    """
    removed: Dict[str, bool] = {}
    _, _, live_path, staging_path = _paths_for_camera(camera_id)
    targets = [live_path, staging_path] if delete_staging else [live_path]
    for p in targets:
        try:
            if os.path.exists(p):
                os.remove(p)
                removed[p] = True
            else:
                removed[p] = False
        except Exception:
            removed[p] = False
    return removed
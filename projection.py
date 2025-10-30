from __future__ import annotations
import json, os, math, re
from dataclasses import dataclass, asdict
from typing import Optional as _OptionalBool
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import cv2

_C_MPS = 299792458.0  # speed of light (m/s)

# ---------- Storage helpers (mirror calibration.py layout) ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CAL_DIR    = os.path.join(BASE_DIR, "calibration")
GLOBAL_MODEL_PATH = os.path.join(CAL_DIR, "camera_model.json")

def _ensure_cal_dirs():
    os.makedirs(CAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(CAL_DIR, "cams"), exist_ok=True)

def _slug_camera_id(camera_id: Optional[str]) -> Optional[str]:
    if not camera_id:
        return None
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(camera_id).strip())
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    return s or None

def _model_path_for(camera_id: Optional[str]) -> str:
    """
    Per-camera live model path. If camera_id is None → legacy global model path.
    """
    _ensure_cal_dirs()
    slug = _slug_camera_id(camera_id)
    if not slug:
        return GLOBAL_MODEL_PATH
    cam_dir = os.path.join(CAL_DIR, "cams", slug)
    os.makedirs(cam_dir, exist_ok=True)
    return os.path.join(cam_dir, "camera_model.json")

def available_camera_models() -> List[str]:
    """
    List camera slugs that have a per-camera model present.
    (Does not include 'legacy'; check GLOBAL_MODEL_PATH separately if needed.)
    """
    _ensure_cal_dirs()
    out: List[str] = []
    cams_dir = os.path.join(CAL_DIR, "cams")
    for name in sorted(os.listdir(cams_dir)):
        p = os.path.join(cams_dir, name, "camera_model.json")
        if os.path.isfile(p):
            out.append(name)
    return out

# ---------- Math helpers ----------
def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

def rotx(a):  # radians
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float64)

def roty(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float64)

def rotz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float64)

def rpy_to_R(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """
    Compose rotation matrix from intrinsic Z(=yaw) → X(=pitch) → Y(=roll) rotations.
    Result maps vectors from the source frame into the rotated frame.
    """
    # Convention: first yaw about Z, then pitch about X, then roll about Y.
    # This is consistent with how we want to map RADAR → PTZ given small misalignments.
    return rotz(yaw_rad) @ rotx(pitch_rad) @ roty(roll_rad)

def normalize_angle_deg(a: float) -> float:
    """Wrap angle to [-180, +180) degrees."""
    a = float(a)
    while a >= 180.0: a -= 360.0
    while a < -180.0: a += 360.0
    return a

# ---------- Camera intrinsics ----------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: Tuple[float, float, float, float, float] = (0,0,0,0,0)  # k1,k2,p1,p2,k3

    @staticmethod
    def from_fov(res_w: int, res_h: int, fov_h_deg: float) -> "Intrinsics":
        # pinhole: fx = (W/2) / tan(FOV_h/2). Keep square pixels unless aspect says otherwise.
        fx = (res_w / 2.0) / math.tan(deg2rad(fov_h_deg) / 2.0)
        fy = fx * (res_w / res_h)  # conservative; you can set fy=fx if sensor has square pixels
        cx, cy = res_w / 2.0, res_h / 2.0
        return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,     0,     1]], dtype=np.float64)

    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64).reshape(-1,1)
    
    def as_tuple(self) -> Tuple[float,float,float,float]:
        """(fx, fy, cx, cy) for quick use."""
        return (self.fx, self.fy, self.cx, self.cy)


# ---------- Extrinsics (radar→camera) ----------
@dataclass
class Extrinsics:
    R_rc: List[List[float]]  # 3x3
    t_rc: List[float]        # 3x1

    def as_np(self):
        return np.array(self.R_rc, dtype=np.float64), np.array(self.t_rc, dtype=np.float64).reshape(3,1)

# ---------- Full model ----------
@dataclass
class CameraModel:
    intr: Intrinsics
    extr: Extrinsics
    meta: dict

    def save(self, path: Optional[str] = None, *,
             camera_id: Optional[str] = None,
             mirror_global: bool = False):
        """
        Save the model either to:
          - explicit 'path' (highest priority),
          - per-camera path resolved via camera_id,
          - or the legacy global path when camera_id is None.
        If mirror_global=True and a per-camera path is used, also write the legacy global file.
        """
        if path is None:
            path = _model_path_for(camera_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "intr": asdict(self.intr),
                "extr": {"R_rc": self.extr.R_rc, "t_rc": self.extr.t_rc},
                "meta": self.meta
            }, f, indent=2)
        if mirror_global and camera_id:
            with open(GLOBAL_MODEL_PATH, "w") as g:
                json.dump({
                    "intr": asdict(self.intr),
                    "extr": {"R_rc": self.extr.R_rc, "t_rc": self.extr.t_rc},
                    "meta": self.meta
                }, g, indent=2)

    @staticmethod
    def load(path: Optional[str] = None, *,
             camera_id: Optional[str] = None,
             fallback_to_global: bool = True) -> "CameraModel":
        """
        Load model from:
          - explicit 'path' (highest priority),
          - per-camera path (if camera_id given),
          - otherwise legacy global path.
        If per-camera not present and fallback_to_global=True, try legacy global.
        """
        if path is None:
            cand = _model_path_for(camera_id)
            if not os.path.isfile(cand) and fallback_to_global:
                cand = GLOBAL_MODEL_PATH
            path = cand
        with open(path, "r") as f:
            d = json.load(f)
        intr = Intrinsics(**d["intr"])
        extr = Extrinsics(**d["extr"])
        return CameraModel(intr=intr, extr=extr, meta=d.get("meta", {}))
    
    @staticmethod
    def try_load(path: Optional[str] = None, *,
                 camera_id: Optional[str] = None,
                 fallback_to_global: bool = True) -> Optional["CameraModel"]:
        """Safe load: return None if file not present/invalid. (Supports per-camera)"""
        try:
            return CameraModel.load(path, camera_id=camera_id, fallback_to_global=fallback_to_global)
        except Exception:
            return None

def try_load_active_model(selected_camera_id: Optional[str] = None) -> Optional[CameraModel]:
    """
    Convenience: prefer the selected camera's model; if unavailable, fall back to the legacy global.
    """
    m = CameraModel.try_load(camera_id=selected_camera_id, fallback_to_global=True)
    return m

# ---------- PTZ mount (RADAR → PTZ) ----------
@dataclass
class PTZMount:
    """
    Rigid transform from RADAR frame to PTZ frame.
    Translation is from RADAR origin to PTZ origin, expressed in RADAR frame.
    Orientation is small misalignment between frames (yaw,pitch,roll in degrees).
    Sign conventions:
      • +X = right, +Y = forward, +Z = up (both frames)
      • yaw: + about Z, pitch: + about X, roll: + about Y
    """
    dx_m: float = 0.0
    dy_m: float = 0.0
    dz_m: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    @staticmethod
    def from_dict(d: Union[Dict[str, Any], "PTZMount", None]) -> "PTZMount":
        if isinstance(d, PTZMount):
            return d
        d = dict(d or {})
        return PTZMount(
            dx_m=float(d.get("dx_m", 0.0)),
            dy_m=float(d.get("dy_m", 0.0)),
            dz_m=float(d.get("dz_m", 0.0)),
            yaw_deg=float(d.get("yaw_deg", 0.0)),
            pitch_deg=float(d.get("pitch_deg", 0.0)),
            roll_deg=float(d.get("roll_deg", 0.0)),
        )

    def R_rp(self) -> np.ndarray:
        """Rotation from RADAR to PTZ frame."""
        return rpy_to_R(math.radians(self.yaw_deg),
                        math.radians(self.pitch_deg),
                        math.radians(self.roll_deg))

    def t_rp(self) -> np.ndarray:
        """Translation from RADAR → PTZ, expressed in RADAR frame, as (3,1)."""
        return np.array([[self.dx_m], [self.dy_m], [self.dz_m]], dtype=np.float64)

# ---------- Radar spherical → radar 3D ----------
def radar_spherical_to_cart(r_m: float, az_deg: float, el_deg: float) -> np.ndarray:
    az, el = deg2rad(az_deg), deg2rad(el_deg)
    x = r_m * math.sin(az) * math.cos(el)
    y = r_m * math.cos(az) * math.cos(el)
    z = r_m * math.sin(el)
    return np.array([x, y, z], dtype=np.float64).reshape(3,1)

# ---------- RADAR → PTZ transform & angles ----------
def radar_to_ptz_point(P_r: np.ndarray, mount: Union[PTZMount, Dict[str, Any], None]) -> np.ndarray:
    """
    Transform a point from RADAR frame to PTZ frame using the mount (R_rp, t_rp).
    P_r: (3,1) in RADAR frame.
    Returns P_p: (3,1) in PTZ frame.
    """
    m = PTZMount.from_dict(mount)
    R = m.R_rp()
    t = m.t_rp()
    # Point in PTZ frame: P_p = R * (P_r - origin_r_to_p_in_radar)
    return R @ (P_r - t)

def radar_obj_to_ptz_angles(r_m: float, az_deg: float, el_deg: float,
                            mount: Union[PTZMount, Dict[str, Any], None]) -> Tuple[float, float]:
    """
    Convert a radar detection (range, az, el) to PTZ pan/tilt angles (deg),
    accounting for radar→PTZ offsets and misalignment.
    Conventions:
      • pan_deg = atan2(x_p, y_p)
      • tilt_deg = atan2(z_p, sqrt(x_p^2 + y_p^2))
      (PTZ +Z is up, +Y is forward, +X is right)
    """
    # Radar point
    P_r = radar_spherical_to_cart(max(float(r_m), 1e-6), float(az_deg), float(el_deg))
    # Transform to PTZ frame
    P_p = radar_to_ptz_point(P_r, mount).reshape(3,)
    x, y, z = float(P_p[0]), float(P_p[1]), float(P_p[2])
    # Horizontal distance safeguard
    rho = math.sqrt(max(x*x + y*y, 0.0))
    if rho < 1e-6:
        # Directly above/below the PTZ origin: pan undefined; choose 0
        pan = 0.0
    else:
        pan = math.degrees(math.atan2(x, y))
    # Tilt up positive; if target below, tilt is negative
    tilt = math.degrees(math.atan2(z, rho))
    return normalize_angle_deg(pan), normalize_angle_deg(tilt)

# ---------- Transform + project ----------
def radar_to_camera(P_r: np.ndarray, extr: Extrinsics) -> np.ndarray:
    R, t = extr.as_np()
    return R @ P_r + t  # (3,1) in camera frame

def project_cam(P_c: np.ndarray, intr: Intrinsics, dist: Optional[np.ndarray]=None) -> Tuple[float,float]:
    # Use cv2.projectPoints for consistent distortion handling
    P_c = np.array(P_c, dtype=np.float64).reshape(3, 1)
    # Guard: points behind camera produce nonsense; let caller decide how to handle None
    if P_c[2, 0] <= 1e-9:
        # Return NaNs so upstream UI can skip this point gracefully
        return float("nan"), float("nan")
    K = intr.K()
    distc = intr.dist_coeffs() if dist is None else dist
    rvec = np.zeros((3,1), dtype=np.float64)
    tvec = np.zeros((3,1), dtype=np.float64)
    pts, _ = cv2.projectPoints(P_c.reshape(1,3), rvec, tvec, K, distc)
    u, v = pts[0,0,0], pts[0,0,1]
    return float(u), float(v)

def project_radar_point_to_pixel(r_m: float, az_deg: float, el_deg: float, model: CameraModel) -> Tuple[float,float]:
    P_r = radar_spherical_to_cart(r_m, az_deg, el_deg)
    P_c = radar_to_camera(P_r, model.extr)
    return project_cam(P_c, model.intr)

# ---------- Pixel → ray / azimuth helpers (for guided calibration UI) ----------
def pixel_to_ray(u: float, v: float, intr: Intrinsics, dist: Optional[Union[np.ndarray, Tuple[float,...]]]=None) -> np.ndarray:
    """
    Convert a pixel to a unit direction vector (ray) in the camera frame.
    If distortion is provided (or in intr.dist), we undistort first.
    """
    fx, fy, cx, cy = intr.as_tuple()
    if dist is None:
        dist = intr.dist_coeffs()
    else:
        dist = np.array(dist, dtype=np.float64).reshape(-1,1)
    K = intr.K()
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K)  # back in pixel coords w.r.t K
    x = (und[0,0,0] - cx) / fx
    y = (und[0,0,1] - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / (np.linalg.norm(ray) + 1e-9)

def pixel_to_approx_azimuth(u: float, intr: Intrinsics) -> float:
    """
    Quick-n-dirty horizontal azimuth (deg) from a pixel.
    Uses pinhole small-angle: az ≈ atan2((u-cx), fx).
    Good enough to choose the nearest radar candidate.
    """
    fx, _, cx, _ = intr.as_tuple()
    return rad2deg(math.atan2((u - cx), fx))

# ---------- Solve extrinsics via PnP (radar 3D ↔ image 2D) ----------
def solve_extrinsics_pnp(world_pts_radar: np.ndarray,
                         img_pts: np.ndarray,
                         intr: Intrinsics,
                         use_ransac=True) -> Tuple[Extrinsics, float]:
    """
    Robust PnP with filtering + fallbacks.
    Tries: RANSAC(EPNP) → refine, then RANSAC(AP3P) → refine, then EPNP→ITERATIVE.
    Returns (Extrinsics, median_reproj_error_px).
    """
    import numpy as _np, cv2 as _cv2

    # ---- sanitize & filter pairs ----
    W = _np.asarray(world_pts_radar, _np.float64).reshape(-1, 3)
    I = _np.asarray(img_pts, _np.float64).reshape(-1, 2)
    mask = _np.all(_np.isfinite(W), axis=1) & _np.all(_np.isfinite(I), axis=1)
    W, I = W[mask], I[mask]
    if I.size:
        # drop near-duplicate pixels (≤0.5 px bins) to avoid degenerate sets
        keep = _np.unique(_np.round(I, 0), axis=0, return_index=True)[1]
        W, I = W[keep], I[keep]
    if W.shape[0] < 6:
        raise RuntimeError(f"Need at least 6 good pairs after filtering, have {W.shape[0]}")

    K, dist = intr.K(), intr.dist_coeffs()

    def _median_err(rvec, tvec, inliers):
        proj, _ = _cv2.projectPoints(W[inliers], rvec, tvec, K, dist)
        e = _np.linalg.norm(proj.reshape(-1, 2) - I[inliers], axis=1)
        return float(_np.median(e))

    best = None  # (med_err, rvec, tvec, inliers_idx)

    # 1) RANSAC + EPNP (looser threshold)
    try:
        ok, rvec, tvec, inl = _cv2.solvePnPRansac(
            W, I, K, dist,
            flags=_cv2.SOLVEPNP_EPNP,
            reprojectionError=12.0, iterationsCount=1000, confidence=0.999
        )
        if ok and inl is not None and len(inl) >= 6:
            ok, rvec, tvec = _cv2.solvePnP(
                W[inl[:, 0]], I[inl[:, 0]], K, dist,
                rvec, tvec, useExtrinsicGuess=True, flags=_cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                best = (_median_err(rvec, tvec, inl[:, 0]), rvec, tvec, inl[:, 0])
    except _cv2.error:
        pass

    # 2) RANSAC + AP3P
    if best is None:
        try:
            ok, rvec, tvec, inl = _cv2.solvePnPRansac(
                W, I, K, dist,
                flags=_cv2.SOLVEPNP_AP3P,
                reprojectionError=12.0, iterationsCount=1000, confidence=0.999
            )
            if ok and inl is not None and len(inl) >= 6:
                ok, rvec, tvec = _cv2.solvePnP(
                    W[inl[:, 0]], I[inl[:, 0]], K, dist,
                    rvec, tvec, useExtrinsicGuess=True, flags=_cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    best = (_median_err(rvec, tvec, inl[:, 0]), rvec, tvec, inl[:, 0])
        except _cv2.error:
            pass

    # 3) Plain EPNP → ITERATIVE (no RANSAC)
    if best is None:
        ok, rvec, tvec = _cv2.solvePnP(W, I, K, dist, flags=_cv2.SOLVEPNP_EPNP)
        if ok:
            ok, rvec, tvec = _cv2.solvePnP(
                W, I, K, dist, rvec, tvec, useExtrinsicGuess=True, flags=_cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                inl = _np.arange(W.shape[0])
                best = (_median_err(rvec, tvec, inl), rvec, tvec, inl)

    if best is None:
        raise RuntimeError("PnP failed on all strategies")

    med_err, rvec, tvec, inl = best
    R, _ = _cv2.Rodrigues(rvec)
    extr = Extrinsics(R_rc=R.tolist(), t_rc=tvec.reshape(3).tolist())
    return extr, float(med_err)

# ---------- Speed correction ----------
_COS_GUARD_DEFAULT = 0.50     # if |cos(theta)| < guard, don't boost (side-look / unreliable geometry)
_MAX_BOOST_DEFAULT = 1.8      # never scale more than this factor
_RADIAL_FLOOR_DEFAULT = 0.20

def _strict_radial_only_enabled() -> bool:
    try:
        return bool(int(os.getenv("PROJ_STRICT_RADIAL_ONLY", "0")))
    except Exception:
        return False

def doppler_hz_to_radial_ms(doppler_hz: float, *, fc_hz: float = 60.75e9) -> float:
    """
    Convert Doppler frequency (Hz) to **radial** speed (m/s) using v = (λ/2) * f_d.
    Pass your radar center frequency via fc_hz if not 60.75 GHz.
    """
    try:
        lam = _C_MPS / float(fc_hz)
        return float(abs(doppler_hz)) * (lam * 0.5)
    except Exception:
        return 0.0

def correct_speed_radial_to_ground(v_radial_mps: float,
                                   r_m: float, az_deg: float, el_deg: float,
                                   model: "CameraModel",
                                   road_dir_cam: Optional[np.ndarray] = None,
                                   *,
                                   strict_radial_only: _OptionalBool[bool] = None) -> float:
    """
    Safe conversion from radial speed → ground/along-road speed.
    Bounded amplification + geometry guards to prevent blow-ups when LOS is
    near-orthogonal to the road direction or elevations are high.
    """
    # ---- sanity ----
    try:
        v = float(v_radial_mps)
    except Exception:
        return 0.0
    if not np.isfinite(v) or v <= 0.0:
        return float(max(v, 0.0))
    if strict_radial_only is None:
        strict_radial_only = _strict_radial_only_enabled()
    if strict_radial_only:
        return float(max(v, 0.0))
    try:
        P_c = radar_to_camera(radar_spherical_to_cart(r_m, az_deg, el_deg), model.extr).reshape(3,)
    except Exception:
        return float(v)  # if transform fails, do not “correct”
    nP = float(np.linalg.norm(P_c))
    if not np.isfinite(nP) or nP < 1e-6:
        return float(v)

    # ---- configuration (env-tunable) ----
    # minimum allowed |cos| (below this → no correction)
    COS_GUARD = float(os.getenv("PROJ_COS_GUARD", "0.5"))          # max ×2 gain
    # cap any amplification even above guard
    MAX_GAIN  = float(os.getenv("PROJ_MAX_GAIN", "1.8"))           # ≤ ×1.8
    # skip correction for steep rays (radar-camera geometry unreliable)
    ELEV_GUARD_DEG = float(os.getenv("PROJ_ELEV_GUARD_DEG", "12"))

    if abs(float(el_deg)) > ELEV_GUARD_DEG:
        return float(v)

    los = P_c / nP
    if road_dir_cam is None:
        # Assume camera forward (Z+) is down-road if not provided.
        road_dir_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    road_dir_cam = road_dir_cam / (np.linalg.norm(road_dir_cam) + 1e-9)

    cos_phi = float(abs(np.dot(los, road_dir_cam)))
    if not np.isfinite(cos_phi) or cos_phi < COS_GUARD:
        return float(v)  # geometry too oblique → don’t upscale

    gain = min(1.0 / max(cos_phi, 1e-6), MAX_GAIN)
    return float(v * gain)

def correct_speed_radial_to_ground_safe(
    *,
    v_radial_ms: float,
    road_axis_unit: Optional[Union[List[float], np.ndarray]] = None,
    obj_vel_vec: Optional[Union[List[float], np.ndarray]] = None,
    cos_guard: float = _COS_GUARD_DEFAULT,
    max_scale: float = 1.8,
    radial_floor_ms: float = _RADIAL_FLOOR_DEFAULT,
    prefer_vector_when_radial_low: bool = True,
    strict_radial_only: _OptionalBool[bool] = None,
) -> float:
    """
    Safe helper (no camera model required):
      • If Doppler radial is below a small floor, **prefer** the object's track velocity
        projected onto the road axis (so you still get non-zero speed for moving targets
        with 0 Hz Doppler).
      • Otherwise, cosine-correct the Doppler radial using the track heading vs. road axis,
        clamped by `max_scale`.
    Returns a non-negative speed (m/s).
    """
    try:
        v = float(abs(v_radial_ms))
        if not np.isfinite(v):
            return 0.0

        if strict_radial_only is None:
            strict_radial_only = _strict_radial_only_enabled()
        if strict_radial_only:
            return v

        # road axis (unit)
        road = np.asarray(
            road_axis_unit if road_axis_unit is not None else [0.0, 1.0, 0.0],
            dtype=np.float64,
        ).reshape(3,)
        road /= (np.linalg.norm(road) + 1e-12)

        # object velocity vector
        vel = np.asarray(
            obj_vel_vec if obj_vel_vec is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        ).reshape(3,)
        vn = float(np.linalg.norm(vel))

        # If Doppler is tiny but we have a track velocity, use its along-road component.
        if prefer_vector_when_radial_low and (v < float(radial_floor_ms)) and (vn > 1e-3):
            # along-road component = |vel ⋅ road|
            return float(abs(np.dot(vel, road)))

        # Otherwise, use Doppler radial with heading-based cosine correction.
        if vn <= 1e-12:
           return v  # no heading → no boost
        vel /= vn
        c = float(abs(np.dot(vel, road)))  # |cos(angle between vel and road)|
        if not np.isfinite(c) or c < float(cos_guard):
            return v
        return float(v * min(1.0 / max(c, 1e-6), float(max_scale)))
    except Exception:
        try:
            return float(abs(v_radial_ms))
        except Exception:
            return 0.0

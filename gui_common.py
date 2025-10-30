# Demo Names
DEMO_NAME_OOB = 'SDK Out of Box Demo'
DEMO_NAME_3DPC = '3D People Counting'
DEMO_NAME_VITALS = 'Vital Signs with People Tracking'
DEMO_NAME_LRPD = 'Long Range People Detection'
DEMO_NAME_MT = 'Mobile Tracker'
DEMO_NAME_SOD = 'Small Obstacle Detection'

# Different methods to color the points 
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'


# Com Port names
CLI_XDS_SERIAL_PORT_NAME = 'XDS110 Class Application/User UART'
DATA_XDS_SERIAL_PORT_NAME = 'XDS110 Class Auxiliary Data Port'
CLI_SIL_SERIAL_PORT_NAME = 'Enhanced COM Port'
DATA_SIL_SERIAL_PORT_NAME = 'Standard COM Port'


# Configurables
MAX_POINTS = 1000
MAX_PERSISTENT_FRAMES = 10

MAX_VITALS_PATIENTS = 2
NUM_FRAMES_PER_VITALS_PACKET = 15
NUM_VITALS_FRAMES_IN_PLOT = 150
NUM_HEART_RATES_FOR_MEDIAN = 10


# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253 # Point not associated, SNR too weak
TRACK_INDEX_BOUNDS = 254 # Point not associated, located outside boundary of interest
TRACK_INDEX_NOISE = 255 # Point not associated, considered as noise


# Defined TLV's
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS                      = 1
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE                        = 2
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE                        = 3
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP               = 4
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP               = 5
MMWDEMO_OUTPUT_MSG_STATS                                = 6
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO            = 7
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP     = 8
MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS                    = 9

MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS                     = 1000
MMWDEMO_OUTPUT_MSG_RD_HEATMAP_LITE                      = 7001
MMWDEMO_OUTPUT_MSG_RANGE_AZIMUTH_STATIC_HEAT_MAP        = 1001
MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST           = 1010
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX             = 1011
MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT            = 1012
MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS                    = 1020
MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION                 = 1021
MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE              = 1030

MMWDEMO_OUTPUT_MSG_VITALSIGNS                           = 1040

# ──────────────────────────────────────────────────────────────────────────────
# Firmware inference helpers 
# ──────────────────────────────────────────────────────────────────────────────
# TLVs that strongly indicate 3D People Counting builds
FW_3DPC_TLVS = {
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST,
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX,
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT,
    MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE,
}

# TLVs commonly seen in Traffic / SDK-OOB style builds
FW_TRAFFIC_HINT_TLVS = {
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP,
    MMWDEMO_OUTPUT_MSG_RANGE_AZIMUTH_STATIC_HEAT_MAP,
    MMWDEMO_OUTPUT_MSG_RD_HEATMAP_LITE,
}

# Point-producing TLVs (lets downstream synthesize tracks if none provided)
POINT_TLVS = {
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS,
    MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS,
    MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS,
}

def infer_fw_mode_from_tlvs(tlv_types: set[int]) -> str:
    """
    Heuristic firmware tag from a set of TLV type IDs seen in a frame/epoch.
    Returns: "3dpc", "traffic", or "unknown".
    """
    if not tlv_types:
        return "unknown"
    # Strong signal: explicit 3D tracker list present → 3DPC
    if MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST in tlv_types:
        return "3dpc"
    # Otherwise bias to traffic if we see classic traffic heatmaps without tracker output
    if tlv_types & FW_TRAFFIC_HINT_TLVS:
        return "traffic"
    # Fallback: unknown (reader may refine over subsequent frames)
    return "unknown"

# ──────────────────────────────────────────────────────────────────────────────
# Firmware-agnostic helpers (optional; used by parser & interface)
# ──────────────────────────────────────────────────────────────────────────────
from math import sqrt, atan2, pi
from typing import Iterable, Dict, Any, Set, Tuple, Optional

# TLV palettes useful to *infer* what firmware likely produced the frame.
# (No branching needed downstream; this is for logging/telemetry only.)

def tlv_types_from_frame(tlvs: Iterable[Dict[str, Any]]) -> Set[int]:
    s: Set[int] = set()
    for t in (tlvs or []):
        try: s.add(int(t.get("type", -1)))
        except Exception: pass
    return s

def spherical_from_xyz(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r = sqrt(x*x + y*y + z*z)
    az = atan2(x, y) * 180.0 / pi
    el = atan2(z, sqrt(x*x + y*y)) * 180.0 / pi
    return float(r), float(az), float(el)

def normalize_speed_mps(v: Optional[float]) -> Optional[float]:
    """Heuristic: >30 is likely km/h → convert to m/s."""
    if v is None: return None
    try:
        vv = float(v)
        return vv/3.6 if vv > 30.0 else vv
    except Exception:
        return None
    
DEFAULT_AZIMUTH_FOV_DEG = 120.0
DEFAULT_RANGE_RES_M     = 0.043
DEFAULT_LANE_BOUNDS = [(-2.0, -0.5), (-0.5, 1.0), (1.0, 2.5)]
TM_VIZ_ENABLED = True

import struct
import sys
import serial
import binascii
import time
import numpy as np
import math

import os
import datetime
from logger import logger
# Local File Imports
from parseTLVs import *
from gui_common import *
from math import sqrt, isfinite

# --- Radar constants (computed from config if available) ---
_C_MPS = 299_792_458.0
try:
    from config_utils import load_config
    _cfg = load_config()
    _fc_ghz = float(_cfg.get("radar", {}).get("center_frequency_ghz", 60.0))
except Exception:
    _fc_ghz = 60.0
_LAMBDA_M = _C_MPS / (_fc_ghz * 1e9)
_DOP_MIN_HZ = 5.0
_DOP_MIN_MPS = 0.05 

def _doppler_to_mps(fd: float) -> float:
    """
    Convert a 'doppler' value to m/s in a firmware-agnostic way.
    Heuristic:
      - If |fd| < 30, it's almost certainly already m/s (OOB 3DPC/Traffic).
      - Otherwise treat as Hz and convert: v = (λ/2) * fd.
    """
    try:
        if not isfinite(fd):
            return 0.0
        a = abs(float(fd))
        if a == 0.0:
            return 0.0
        if a < 30.0:
            return float(fd)                     # m/s
        return float((_LAMBDA_M * fd) / 2.0)     # from Hz → m/s
    except Exception:
        return 0.0

def parseStandardFrame(frameData):
    headerStruct = 'Q8I'
    frameHeaderLen = struct.calcsize(headerStruct)
    tlvHeaderLength = 8

    outputDict = {}
    outputDict['error'] = 0

    try:
        # Read in frame Header
        magic, version, totalPacketLen, platform, frameNum, timeCPUCycles, numDetectedObj, numTLVs, subFrameNum = struct.unpack(headerStruct, frameData[:frameHeaderLen])
    except:
        print('Error: Could not read frame header')
        outputDict['error'] = 1

    # Move frameData ptr to start of 1st TLV    
    frameData = frameData[frameHeaderLen:]

    # Save frame number to output
    outputDict['frameNum'] = frameNum

    # print("")
    # print ("FrameNum: ", frameNum)

    # Initialize the point cloud struct since it is modified by multiple TLV's
    # Each point has the following: X, Y, Z, Doppler, SNR, Noise, Track index
    if numDetectedObj > 500:
        return {"trackData": [], "numDetectedTracks": 0}

    outputDict['pointCloud'] = np.zeros((numDetectedObj, 7), np.float64)
    # Initialize the track indexes to a value which indicates no track
    outputDict['pointCloud'][:, 6] = 255
    seen_tlv_types = set()
    # Find and parse all TLV's
    for i in range(numTLVs):
        try:
            tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
            frameData = frameData[tlvHeaderLength:]
            payload = frameData[:tlvLength]
        except Exception as e:
            outputDict['error'] = 2
            break  # stop parsing further

        # Detected Points
        if (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS): 
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parsePointCloudTLV(payload, tlvLength, outputDict['pointCloud'])
        # Range Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE):
            try:
                num_bins = int(tlvLength / 2)  # 2 bytes per bin
                range_profile = struct.unpack(f"{num_bins}H", payload)
                outputDict['range_profile'] = list(range_profile)
            except Exception as e:
                logger.warning(f"[Parse] Failed RANGE_PROFILE TLV: {e}")
        # Noise Profile
        elif (tlvType == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE):
            try:
                num_bins = int(tlvLength / 2)
                noise_profile = struct.unpack(f"{num_bins}H", payload)
                outputDict['noise_profile'] = list(noise_profile)
            except Exception as e:
                logger.warning(f"[Parse] Failed NOISE_PROFILE TLV: {e}")
        # Range–Doppler Heatmap (Lite, custom: 7001)
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RD_HEATMAP_LITE):
            try:
                # TLV layout: [rows(2), cols(2), offset_q7(2), scale_q7(2)] + rows*cols bytes
                if tlvLength < 8:
                    raise ValueError("RD_HEATMAP_LITE tlv too small")
                rows, cols, off_q7, sc_q7 = struct.unpack('<HHhH', payload[:8])
                pix = payload[8:8 + rows*cols]
                if len(pix) != rows*cols:
                    raise ValueError("RD_HEATMAP_LITE payload length mismatch")
                # Store as uint8 list; the rest of the pipeline will flatten/reshape as needed
                outputDict['range_doppler_heatmap'] = np.frombuffer(pix, dtype=np.uint8).tolist()
                outputDict['range_doppler_meta'] = {
                    "rows": int(rows), "cols": int(cols),
                    "offset_q7": int(off_q7), "scale_q7": int(sc_q7)
                }
            except Exception as e:
                logger.warning(f"[Parse] RD_HEATMAP_LITE failed: {e}")
        # Range Doppler Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP):
            try:
                if tlvLength % 2 != 0:
                    raise ValueError("buffer size must be multiple of 2")
                heatmap = np.frombuffer(payload, dtype=np.int16)
                outputDict['range_doppler_heatmap'] = heatmap.tolist()
            except Exception as e:
                logger.debug(f"[Parse] RD_HEAT_MAP parse skipped: {e}")
        # Static Azimuth / Range-Azimuth Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP) or \
             (tlvType == MMWDEMO_OUTPUT_MSG_RANGE_AZIMUTH_STATIC_HEAT_MAP):
            try:
                buf = payload  # use the sliced payload for this TLV
                if tlvLength % 4 == 0:
                    arr = np.frombuffer(buf, dtype=np.float32)
                    if not np.isfinite(arr).all():
                        arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                elif tlvLength % 2 == 0:
                    # Some firmwares send int16 or interleaved I/Q; we at least get magnitude
                    i16 = np.frombuffer(buf, dtype=np.int16)
                    if i16.size % 2 == 0:
                        re = i16[0::2].astype(np.float32)
                        im = i16[1::2].astype(np.float32)
                        arr = np.sqrt(re*re + im*im)
                    else:
                        arr = i16.astype(np.float32)
                else:
                    raise ValueError("buffer size must be a multiple of element size")
                outputDict['range_azimuth_heatmap'] = arr.tolist()
                logger.info(f"[HEATMAP] Range-Azimuth heatmap len={arr.size}")
            except Exception as e:
                logger.warning(f"[Parse] Failed RANGE_AZIMUTH_HEATMAP TLV: {e}")
        # Performance Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_STATS):
            outputDict['stats'] = parseStatsTLV(payload)
        # Side Info
        elif (tlvType == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO):
            outputDict['pointCloud'] = parseSideInfoTLV(payload, tlvLength, outputDict['pointCloud'])
         # Azimuth Elevation Static Heatmap
        elif (tlvType == MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP):
            try:
                if tlvLength % 4 == 0:
                    arr = np.frombuffer(payload, dtype=np.float32)
                elif tlvLength % 2 == 0:
                    arr = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
                else:
                    raise ValueError("buffer size must be a multiple of 2")
                outputDict['azimuth_elevation_heatmap'] = arr.tolist()
                # also surface under RA key so UI has one place to look
                outputDict.setdefault('range_azimuth_heatmap', arr.tolist())
                logger.info(f"[HEATMAP] Azimuth–Elevation heatmap len={arr.size}")
            except Exception as e:
                logger.warning(f"[Parse] Failed STATIC_HEATMAP TLV: {e}")
        # Temperature Statistics
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS):
            pass
        # Spherical Points
        elif (tlvType == MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseSphericalPointCloudTLV(payload, tlvLength, outputDict['pointCloud'])
        # Target 3D
        elif tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST:
            try:
                track_data = parseTrackTLV(payload, tlvLength)
                outputDict['trackData'] = track_data
                outputDict['numDetectedTracks'] = len(track_data)
                logger.info(f"[parseTrackTLV] Parsed {len(track_data)} target(s)")
            except Exception as e:
                outputDict['trackData'] = []
                outputDict['numDetectedTracks'] = 0
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT):
            outputDict['numDetectedHeights'], outputDict['heightData'] = parseTrackHeightTLV(frameData[:tlvLength], tlvLength)
         # Target index
        elif (tlvType == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX):
            outputDict['trackIndexes'] = parseTargetIndexTLV(frameData[:tlvLength], tlvLength)
         # Capon Compressed Spherical Coordinates
        elif (tlvType == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS):
            outputDict['numDetectedPoints'], outputDict['pointCloud'] = parseCompressedSphericalPointCloudTLV(frameData[:tlvLength], tlvLength, outputDict['pointCloud'])
        # Presence Indication
        elif (tlvType == MMWDEMO_OUTPUT_MSG_PRESCENCE_INDICATION):
            pass
        # Occupancy State Machine
        elif (tlvType == MMWDEMO_OUTPUT_MSG_OCCUPANCY_STATE_MACHINE):
            outputDict['occupancy'] = parseOccStateMachTLV(frameData[:tlvLength])
        elif (tlvType == MMWDEMO_OUTPUT_MSG_VITALSIGNS):
            outputDict['vitals'] = parseVitalSignsTLV(frameData[:tlvLength], tlvLength)
        else:
            outputDict.setdefault('unknown_tlvs', []).append({
                'type': tlvType,
                'length': tlvLength,
                'raw_data': frameData[:tlvLength].hex()[:200]  # store first 100 bytes as hex preview
            })
        seen_tlv_types.add(int(tlvType))
        # print ("Frame Data after tlv parse: ", frameData[:10])
        # Move to next TLV
        frameData = frameData[tlvLength:]
        # print ("Frame Data at end of TLV: ", frameData[:10])

    outputDict['tlv_types'] = list(seen_tlv_types)

    # ─────────────────────────────────────────────────────────────
    # Firmware-agnostic fallbacks (Traffic/OOB & sparse 3DPC)
    # 1) If no native tracks → try building from point cloud
    # 2) If still nothing → promote RA-heatmap peak as a minimal track
    # ─────────────────────────────────────────────────────────────
    if 'trackData' not in outputDict or not outputDict.get('trackData'):
        pc = outputDict.get('pointCloud', None)
        if isinstance(pc, np.ndarray) and pc.size >= 3:
            tracks = _derive_targets_from_pointcloud(pc)
            outputDict['trackData'] = tracks
            outputDict['numDetectedTracks'] = len(tracks)
        else:
            hm_tracks = _derive_target_from_heatmap(outputDict)
            outputDict['trackData'] = hm_tracks
            outputDict['numDetectedTracks'] = len(hm_tracks)

    try:
        # compact point set
        pc = outputDict.get('pointCloud')
        pts = []
        if isinstance(pc, np.ndarray) and pc.ndim == 2 and pc.shape[0] > 0:
            step = max(1, pc.shape[0] // 1500 or 1)
            # keep first 5 cols: x,y,z,doppler,snr (if present)
            keep = min(pc.shape[1], 5)
            pts = pc[::step, :keep].astype(float).tolist()

        tracks = outputDict.get('trackData', []) or []

        # Per-lane counts (Y forward; X bins from gui_common)
        from gui_common import DEFAULT_LANE_BOUNDS, DEFAULT_AZIMUTH_FOV_DEG, TM_VIZ_ENABLED
        lane_counts = [0]*len(DEFAULT_LANE_BOUNDS)
        for t in tracks:
            x = float(t.get("posX", t.get("x", 0.0)))
            y = float(t.get("posY", t.get("y", 0.0)))
            if y < 0:  # behind radar → ignore for lanes
                continue
            for i,(xmin,xmax) in enumerate(DEFAULT_LANE_BOUNDS):
                if xmin <= x <= xmax:
                    lane_counts[i] += 1
                    break

        tm_payload = {
            "xy": {"points": pts, "tracks": tracks},
            "gate": {"tracks": tracks},  # gating/ellipses can be drawn in UI
            "fov_deg": float(DEFAULT_AZIMUTH_FOV_DEG),
            "stats": {
                "frame": int(outputDict.get("frameNum", 0)),
                "lanes": {"current": lane_counts}
            }
        }
        if TM_VIZ_ENABLED:
            outputDict["ui_mode"] = "tm"
            outputDict["tm_viz"] = tm_payload
            # If UI still checks for heatmap presence, explicitly clear it
            outputDict.pop("range_azimuth_heatmap", None)
            outputDict.pop("range_doppler_heatmap", None)
    except Exception as _e:
        logger.debug(f"[tm_viz] build skipped: {_e}")

    return outputDict

# ─────────────────────────────────────────────────────────────────
# Simple, dependency-free clustering to build track-like objects
# from OOB/Traffic point clouds.
# pc layout: [x, y, z, doppler, snr, noise, trackIdx]
# ─────────────────────────────────────────────────────────────────
def _derive_targets_from_pointcloud(pc: np.ndarray):
    try:
        if pc.ndim != 2 or pc.shape[1] < 4 or pc.shape[0] == 0:
            return []
        # 0.45 m voxel binning (approx DBSCAN eps without sklearn)
        vx = np.floor(pc[:, 0] / 0.45)
        vy = np.floor(pc[:, 1] / 0.45)
        vz = np.floor(pc[:, 2] / 0.60)  # looser on Z
        keys = (vx.astype(np.int64) << 42) ^ (vy.astype(np.int64) << 21) ^ vz.astype(np.int64)
        # group indices by key
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, k in enumerate(keys):
            groups[int(k)].append(idx)

        tracks = []
        tid_seq = 1
        for _, idxs in groups.items():
            if len(idxs) < 3:   # noise gate
                continue
            pts = pc[idxs]
            # centroid position
            pos = pts[:, :3].mean(axis=0)
            fd = float(np.median(pts[:, 3])) if pts.shape[1] > 3 else 0.0
            v_mps = _doppler_to_mps(fd)
            # Accept if either m/s or Hz gate would pass
            has_motion = (abs(v_mps) >= _DOP_MIN_MPS) or (abs(fd) >= _DOP_MIN_HZ)
            # Project radial velocity along the centroid ray to get a vector
            dist = float(sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2))
            if has_motion and dist > 1e-6:
                ux, uy, uz = pos[0]/dist, pos[1]/dist, pos[2]/dist
                velX, velY, velZ = v_mps * ux, v_mps * uy, v_mps * uz
                dop_hz = float((2.0 * v_mps) / _LAMBDA_M) if abs(v_mps) >= _DOP_MIN_MPS else 0.0
            else:
                velX = velY = velZ = 0.0
                dop_hz = 0.0
            snr = float(np.median(pts[:, 4])) if pts.shape[1] > 4 else 0.0
            noise = float(np.median(pts[:, 5])) if pts.shape[1] > 5 else 0.0
            dist  = float(sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2))

            tracks.append({
                "tid": int(tid_seq),
                "posX": float(pos[0]),
                "posY": float(pos[1]),
                "posZ": float(pos[2]),
                "velX": float(velX), "velY": float(velY), "velZ": float(velZ),
                "accX": 0.0, "accY": 0.0, "accZ": 0.0,
                "g": 0.0,
                "confidence": 0.5,
                "snr": snr,
                "noise": noise,
                # canonical speed fields used by pipeline
                "speed_kmh": float(abs(v_mps) * 3.6),
                "velocity": float(v_mps),
                "doppler_hz": dop_hz,
                "velocity_source": ("doppler_pc" if has_motion else "none"),
                "origin": "pc_cluster",
                "distance": dist,
            })
            tid_seq += 1
        return tracks
    except Exception:
        return []

def _derive_target_from_heatmap(outputDict: dict):
    """
    When a firmware/frame carries only heatmaps (no points/tracks),
    synthesize ONE minimal target from the max RA heatmap bin.
    """
    ra = outputDict.get("range_azimuth_heatmap")
    if ra is None or not hasattr(ra, "shape"):
        return []
    try:
        # Resolve indices of global maximum
        idx = np.argmax(ra)
        rbin, abin = np.unravel_index(int(idx), ra.shape)
        H, W = ra.shape  # H=range bins, W=angle bins
        # Pull resolution if stats exist; otherwise use safe defaults
        stats = outputDict.get("stats", {})
        rng_res = float(stats.get("rangeResolutionMeters", 0.043))  # ~4.3 cm default
        # Map angle bin → azimuth degrees. Default ±60° across W bins.
        fov_deg = float(stats.get("azimuthFovDeg", 120.0))
        az_deg = ( (abin / max(W-1,1)) - 0.5 ) * fov_deg
        dist_m = (rbin + 0.5) * rng_res

        # Place a target in polar → Cartesian (z=0)
        az_rad = np.deg2rad(az_deg)
        x = dist_m * np.sin(az_rad)
        y = dist_m * np.cos(az_rad)
        snr = float(ra[rbin, abin])
        return [{
            "tid": 1,
            "posX": float(x), "posY": float(y), "posZ": 0.0,
            "velX": 0.0, "velY": 0.0, "velZ": 0.0,
            "accX": 0.0, "accY": 0.0, "accZ": 0.0,
            "g": 0.0, "confidence": 0.25,
            "snr": snr, "noise": 0.0,
            "speed_kmh": 0.0,
            "velocity": 0.0,
            "doppler_hz": 0.0,
            "distance": float(dist_m),
            "velocity_source": "none",
            "origin": "heatmap_peak",
        }]
    except Exception:
        return []

# Capon Processing Chain uses a modified header with a slightly different set of TLV's, so it needs its own frame parser
# def parseCaponFrame(frameData):
#     tlvHeaderLength = 8
#     headerLength = 48
#     headerStruct = 'Q9I2H'
    
#     outputDict = {}
#     outputDict['error'] = 0

#     try:
#         magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum =  struct.unpack(headerStruct, frameData[:headerLength])
#     except Exception as e:
#         print('Error: Could not read frame header')
#         outputDict['error'] = 1

#     outputDict['frameNum'] = frameNum        
#     frameData = frameData[headerLength:]
#     # Check TLVs
#     for i in range(numTLVs):
#         #try:
#         #print("DataIn Type", type(dataIn))
#         try:
#             tlvType, tlvLength = tlvHeaderDecode(frameData[:tlvHeaderLength])
#             frameData = frameData[tlvHeaderLength:]
#             dataLength = tlvLength - tlvHeaderLength
#         except:
#             print('TLV Header Parsing Failure')
#             outputDict['error'] = 2
        
#         # OOB Point Cloud
#         if (tlvType == 1): 
#             pass
#         # Range Profile
#         elif (tlvType == 2):
#             pass
#         # Noise Profile
#         elif (tlvType == 3):
#             pass
#         # Static Azimuth Heatmap
#         elif (tlvType == 4):
#             pass
#         # Range Doppler Heatmap
#         elif (tlvType == 5):
#             pass
#         # Capon Polar Coordinates
#         elif (tlvType == 6):
#             numDetectedPoints, parsedPointCloud = parseCaponPointCloudTLV(frameData[:dataLength], dataLength)
#             outputDict['pointCloudCapon'] = parsedPointCloud
#             outputDict['numDetectedPoints'] = numDetectedPoints
#         # Target 3D
#         elif (tlvType == 7):
#             numDetectedTracks, parsedTrackData = parseTrackTLV(frameData[:dataLength], dataLength)
#             outputDict['trackData'] = parsedTrackData
#             outputDict['numDetectedTracks'] = numDetectedTracks
#          # Target index
#         elif (tlvType == 8):
#             #self.parseTargetAssociations(dataIn[:dataLength])
#             outputDict['trackIndexCapon'] = parseTargetIndexTLV(frameData[:dataLength], dataLength)
#         # Classifier Output
#         elif (tlvType == 9):
#             pass
#         # Stats Info
#         elif (tlvType == 10):
#             pass
#         # Presence Indicator
#         elif (tlvType == 11):
#             pass
#         else:
#             print ("Warning: invalid TLV type: %d" % (tlvType))
        
#         frameData = frameData[dataLength:]
#     return outputDict



# Decode TLV Header
def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('2I', data)
    return tlvType, tlvLength

